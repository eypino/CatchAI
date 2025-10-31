import sounddevice as sd
import numpy as np
import pandas as pd
import queue
import threading
import webrtcvad
from faster_whisper import WhisperModel
import string
import os
import json
import faiss
from openai import OpenAI
import re
import logging
import time
from collections import deque
from config_github_token import obtener_token

# =================================================================
# BLOQUE 0: CONFIGURACIÓN Y LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)

def load_config():
    """Carga la configuración desde config.json o usa valores por defecto."""
    try:
        # Asegúrate que la ruta al config sea correcta desde donde ejecutas el script
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
            logging.info("Archivo config.json cargado exitosamente.")
            return json.load(f)
    except FileNotFoundError:
        logging.warning("No se encontró config.json. Usando configuración por defecto.")
        return {
          "audio": {"samplerate": 16000, "block_duration_ms": 500, "chunk_duration_ms": 2000, "channels": 1, "sentence_pause_ms": 1200},
          "vad": {"sensitivity": 2, "speech_threshold": 0.3},
          "whisper": {"model_size": "small", "device": "cpu", "compute_type": "int8", "beam_size": 1, "no_speech_prob": 0.6},
          "faiss": {"top_k": 20, "similarity_threshold": 0.35},
          "openai": {"embedding_model": "text-embedding-3-small", "chat_model": "gpt-4o-mini"}
        }

CONFIG = load_config()

# ==== Configuración de audio (desde config.json) ====
samplerate = CONFIG['audio']['samplerate']
block_duration = CONFIG['audio']['block_duration_ms'] / 1000.0
chunk_duration = CONFIG['audio']['chunk_duration_ms'] / 1000.0
channels = CONFIG['audio']['channels']
sentence_pause_ms = CONFIG['audio']['sentence_pause_ms']

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

# ==== Colas ====
audio_queue = queue.Queue()
segment_queue = queue.Queue()
text_queue = queue.Queue()
resultado_queue = queue.Queue()

audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo Whisper ====
model_size = CONFIG['whisper']['model_size']
logging.info(f"Cargando modelo Whisper '{model_size}' en dispositivo '{CONFIG['whisper']['device']}'...")
model = WhisperModel(model_size, device=CONFIG['whisper']['device'], compute_type=CONFIG['whisper']['compute_type'])
logging.info("✅ Modelo Whisper cargado.")

# ==== Detector de voz (VAD) ====
vad = webrtcvad.Vad(CONFIG['vad']['sensitivity'])

# ==== Rutas ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")

GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_definitivo.csv")
TRANSCRIPCION_TXT = os.path.join(STORAGE_DIR, "transcripciones.txt")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")
PRONOMBRES_PATH = os.path.join(STORAGE_DIR, "Pronombres_map.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")

# ==== Cliente OpenAI (Azure/GitHub) ====
GITHUB_TOKEN = obtener_token()
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=GITHUB_TOKEN
)

# ==== Inicializar archivos ====
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Pronombres ====
try:
    with open(PRONOMBRES_PATH, "r", encoding="utf-8") as f:
        PRONOMBRES_MAP = json.load(f)
except Exception:
    PRONOMBRES_MAP = {
        "yo": "YO", "tú": "TÚ", "usted": "USTED", "él": "ÉL", "ella": "ELLA",
        "nosotros": "NOSOTROS", "ustedes": "USTEDES", "ellos": "ELLOS", "ellas": "ELLAS"
    }
    logging.warning("Pronombres JSON no encontrado — usando mapa por defecto.")

# ==== Funciones de marcadores ====
def marcar_pronombres(texto):
    palabras = texto.lower().split()
    marcadas = []
    for p in palabras:
        p_clean = p.translate(str.maketrans('', '', string.punctuation))
        if p_clean in PRONOMBRES_MAP:
            marcadas.append(f"[PRON:{PRONOMBRES_MAP[p_clean]}]")
        else:
            marcadas.append(p)
    return " ".join(marcadas)

def quitar_marcadores(texto):
    return re.sub(r"\[PRON:[^\]]+\]\s*", "", texto).strip()

# =================================================================
# BLOQUE 1: CARGA Y CREACIÓN DE FAISS
# =================================================================
logging.info("🧠 Cargando glosario y FAISS...")
glosario_df = pd.read_csv(GLOSARIO_PATH)
glosario_df = glosario_df.fillna('')

glosario_df['Texto_Contexto'] = (
    glosario_df['Palabra'].str.upper().str.strip() + ". " +
    "Descripción: " + glosario_df['Descripción'] + ". " +
    "Categoría: " + glosario_df['Categoría'] + ". " +
    "Sinónimos: " + glosario_df['Sinónimos'] + ". " +
    "Antónimos: " + glosario_df['Antónimos']
)
glosario = glosario_df["Palabra"].str.upper().str.strip().tolist()
glosario_textos = glosario_df["Texto_Contexto"].tolist()

if not os.path.exists(FAISS_INDEX_PATH):
    logging.info("⚙️ Creando índice FAISS (primera vez) con contexto usando OpenAI...")
    embeddings = []
    batch_size = 100
    for i in range(0, len(glosario_textos), batch_size):
        batch = glosario_textos[i:i+batch_size]
        resp = client.embeddings.create(model=CONFIG['openai']['embedding_model'], input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    glosario_embeddings = np.array(embeddings).astype("float32")
    np.save(EMB_PATH, glosario_embeddings)
    dim = glosario_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(glosario_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)

logging.info(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")

# =================================================================
# BLOQUE 2: FUNCIÓN DE BÚSQUEDA HÍBRIDA
# =================================================================
def buscar_glosas_hibrido(texto, top_k=CONFIG['faiss']['top_k'], threshold=CONFIG['faiss']['similarity_threshold']):
    texto_para_buscar = quitar_marcadores(texto).strip()
    if not texto_para_buscar:
        return []

    palabras_individuales = list(set(texto_para_buscar.split()))
    textos_a_vectorizar = [texto_para_buscar] + palabras_individuales

    try:
        resp = client.embeddings.create(
            model=CONFIG['openai']['embedding_model'],
            input=textos_a_vectorizar
        )
        embeddings_batch = [d.embedding for d in resp.data]
        embeddings_np = np.array(embeddings_batch).astype("float32")
    except Exception as e:
        logging.error(f"Error generando embeddings para la búsqueda híbrida: {e}")
        return []

    D, I = index.search(embeddings_np, top_k)

    resultados_indices = set()
    for i in range(len(I)):
        max_d, min_d = float(np.max(D[i])), float(np.min(D[i]))
        denom = (max_d - min_d + 1e-6)
        similitudes = 1 - ((D[i] - min_d) / denom)
        for idx, sim in zip(I[i], similitudes):
            if sim >= threshold:
                resultados_indices.add(idx)
    return [glosario[idx] for idx in resultados_indices]


# =================================================================
# BLOQUE 3: TRADUCCIÓN CON MEMORIA Y PROMPT MEJORADO
# =================================================================
def traducir_a_glosas(texto, contexto_anterior=""):
    texto = texto.strip()
    if not texto:
        return []

    candidatas = buscar_glosas_hibrido(texto)
    if not candidatas:
        logging.warning(f"Sin glosas candidatas válidas para: '{texto}'")
        return []

    prompt = f"""
Eres un traductor experto de español a glosas de la Lengua de Señas Chilena (LSCh).

**Contexto de la conversación anterior (si es relevante):**
{contexto_anterior if contexto_anterior else "No hay contexto previo."}

**Texto a traducir:**
{texto}

**Reglas obligatorias:**
1.  **Usa ÚNICAMENTE glosas de la siguiente lista:** {", ".join(candidatas)}
2.  NO inventes, modifiques ni combines glosas. La glosa debe ser exactamente como en la lista.
3.  Si ninguna glosa de la lista aplica para traducir el texto, devuelve una lista vacía.
4.  Ordena las glosas siguiendo la estructura de la LSCh (ej. Sujeto-Objeto-Verbo o Tema-Comentario).
5.  Devuelve la respuesta como un objeto JSON con una única clave "glosas" que contenga una lista de strings.

**Ejemplos de traducción:**
-   Texto: "mi hermana quiere un auto rojo"
-   Glosas candidatas: HERMANA, QUERER, AUTO, ROJO, MÍO, CASA
-   Salida: {{"glosas": ["HERMANA", "MÍO", "AUTO", "ROJO", "QUERER"]}}

-   Texto: "¿dónde trabajas tú?"
-   Glosas candidatas: DÓNDE, TRABAJAR, TÚ, COMER
-   Salida: {{"glosas": ["TÚ", "TRABAJAR", "DÓNDE"]}}

-   Texto: "yo no voy a la fiesta"
-   Glosas candidatas: YO, FIESTA, IR, NO
-   Salida: {{"glosas": ["YO", "FIESTA", "IR", "NO"]}}
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['openai']['chat_model'],
            messages=[
                {"role": "system", "content": "Eres un traductor de texto a glosas de la Lengua de Señas Chilena (LSCh). Solo puedes usar glosas existentes y debes responder en formato JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        content = getattr(response.choices[0].message, "content", None)
        if not content:
            logging.warning("GPT devolvió contenido vacío.")
            return []

        salida = json.loads(content)
        glosas = salida.get("glosas", [])

        if not isinstance(glosas, list):
            logging.warning(f"GPT no devolvió una lista en el JSON: {glosas}")
            return []

        glosas_filtradas = [g.upper() for g in glosas if g and g.upper() in glosario]
        
        seen = set()
        glosas_finales = [g for g in glosas_filtradas if not (g in seen or seen.add(g))]
        
        return glosas_finales

    except Exception as e:
        logging.error(f"Error en la llamada a GPT: {e}")
        return []

# =================================================================
# BLOQUE 4: FLUJO DE AUDIO CON AGREGACIÓN DE FRASES
# =================================================================
def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(status)
    audio_queue.put(indata.copy())

def record_audio():
    global sistema_activo
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, blocksize=frame_per_block):
        logging.info("🎙️ Grabando... Presiona Ctrl+C para detener.")
        while sistema_activo:
            sd.sleep(1000)

def is_speech(audio_data):
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    n_frames = len(audio_int16) // frame_size
    if n_frames == 0:
        return False
    speech_frames = sum(1 for i in range(n_frames) if vad.is_speech(audio_int16[i*frame_size:(i+1)*frame_size].tobytes(), samplerate))
    return (speech_frames / n_frames) > CONFIG['vad']['speech_threshold']

def transcribe_audio():
    global audio_buffer, sistema_activo
    while sistema_activo:
        try:
            block = audio_queue.get(timeout=1)
            audio_buffer = np.vstack((audio_buffer, block))
            while len(audio_buffer) >= frame_per_chunk:
                audio_data = audio_buffer[:frame_per_chunk].flatten().astype(np.float32)
                audio_buffer = audio_buffer[frame_per_chunk:]
                if is_speech(audio_data):
                    segments, _ = model.transcribe(audio_data, beam_size=CONFIG['whisper']['beam_size'], language="es")
                    for segment in segments:
                        if segment.no_speech_prob < CONFIG['whisper']['no_speech_prob'] and segment.text.strip():
                            segment_queue.put(segment.text)
        except queue.Empty:
            continue

def aggregate_sentences():
    global sistema_activo
    sentence_buffer = []
    last_segment_time = time.time()
    pause_threshold = CONFIG['audio']['sentence_pause_ms'] / 1000.0

    while sistema_activo:
        try:
            segment = segment_queue.get(timeout=pause_threshold)
            sentence_buffer.append(segment)
            last_segment_time = time.time()
        except queue.Empty:
            if sentence_buffer:
                full_sentence = " ".join(sentence_buffer).strip()
                logging.info(f"Frase detectada: '{full_sentence}'")
                text_queue.put(full_sentence)
                sentence_buffer = []

def process_text():
    global sistema_activo
    conversational_history = deque(maxlen=2)

    while sistema_activo:
        try:
            texto = text_queue.get(timeout=1)
            texto_marcado = marcar_pronombres(texto)
            contexto_str = " ".join(conversational_history)
            glosas = traducir_a_glosas(texto_marcado, contexto_anterior=contexto_str)
            conversational_history.append(texto)

            resultado = {"texto": texto, "glosas": glosas}
            logging.info(f"Resultado: {resultado}")

            resultado_queue.put(resultado)

            with open(TRANSCRIPCION_TXT, "a", encoding="utf-8") as f:
                f.write(str(resultado) + "\n")
            
            try:
                with open(TRANSCRIPCION_JSON, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(resultado)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except (FileNotFoundError, json.JSONDecodeError):
                 with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
                    json.dump([resultado], f, ensure_ascii=False, indent=4)

        except queue.Empty:
            continue

# =================================================================
# BLOQUE 5: CONTROL DEL SISTEMA Y CIERRE SEGURO
# =================================================================
def iniciar_sistema():
    global sistema_activo
    sistema_activo = True
    
    threads = [
        threading.Thread(target=record_audio, name="AudioRecorder", daemon=True),
        threading.Thread(target=transcribe_audio, name="WhisperTranscriber", daemon=True),
        threading.Thread(target=aggregate_sentences, name="SentenceAggregator", daemon=True),
        threading.Thread(target=process_text, name="TextProcessor", daemon=True)
    ]
    
    for t in threads:
        t.start()
        
    logging.info("🚀 Sistema de transcripción + glosas LSCh iniciado.")

def detener_sistema():
    global sistema_activo
    sistema_activo = False
    logging.info("🛑 Deteniendo sistema... por favor espere.")

if __name__ == "__main__":
    sistema_activo = False
    iniciar_sistema()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detener_sistema()