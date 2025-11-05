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
import asyncio
from collections import deque
from config_github_token import obtener_token
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

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
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
            logging.info("Archivo config.json cargado exitosamente.")
            return json.load(f)
    except FileNotFoundError:
        logging.warning("No se encontró config.json. Usando configuración por defecto.")
        return {
          "audio": {"samplerate": 16000, "block_duration_ms": 500, "chunk_duration_ms": 2500, "channels": 1, "sentence_pause_ms": 1000},
          "vad": {"sensitivity": 2, "speech_threshold": 0.3},
          "whisper": {"model_size": "small", "device": "cpu", "compute_type": "float32", "beam_size": 5, "no_speech_prob": 0.6},
          "faiss": {"top_k": 20, "similarity_threshold": 0.6},
          "openai": {"embedding_model": "text-embedding-3-small", "chat_model": "gpt-4o-mini"}
        }

CONFIG = load_config()

# ==== Configuración de audio ====
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
resultado_queue = asyncio.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)
sistema_activo = False

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
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=GITHUB_TOKEN)

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
# BLOQUE 2 (CORREGIDO): FUNCIÓN DE BÚSQUEDA CONTEXTUAL
# =================================================================
def buscar_candidatas_contextuales(texto, top_k=CONFIG['faiss']['top_k'], threshold=CONFIG['faiss']['similarity_threshold']):
    """
    Busca glosas candidatas basándose en el embedding de la frase COMPLETA
    y devuelve una lista de diccionarios con contexto para el LLM.
    """
    texto_para_buscar = quitar_marcadores(texto).strip()
    if not texto_para_buscar:
        return [] # Devuelve lista vacía

    try:
        # 1. Vectorizar SOLO el texto completo para obtener el contexto general
        resp = client.embeddings.create(
            model=CONFIG['openai']['embedding_model'],
            input=[texto_para_buscar] 
        )
        embedding_query = np.array([resp.data[0].embedding]).astype("float32")
    except Exception as e:
        logging.error(f"Error generando embedding para la consulta: {e}")
        return []

    # 2. Buscar en FAISS
    D, I = index.search(embedding_query, top_k)
    
    resultados = []
    
    # 3. Calcular similitud y filtrar
    indices = I[0]
    distancias = D[0]
    
    # Evitar división por cero si todas las distancias son iguales
    max_d, min_d = float(np.max(distancias)), float(np.min(distancias))
    denom = (max_d - min_d + 1e-6) 
    
    similitudes = 1 - ((distancias - min_d) / denom)
    seen_glosas = set()

    for idx, sim in zip(indices, similitudes):
        if sim >= threshold:
            # 4. Obtener la información COMPLETA de la glosa desde el dataframe
            glosa_palabra = glosario_df.iloc[idx]['Palabra'].upper().strip()
            
            # Evitar duplicados
            if glosa_palabra not in seen_glosas:
                glosa_desc = glosario_df.iloc[idx]['Descripción']
                
                # 5. Devolver un diccionario con contexto
                #    AQUÍ ESTÁ LA CORRECCIÓN: float(sim)
                resultados.append({
                    "glosa": glosa_palabra,
                    "descripcion": glosa_desc,
                    "similitud_score": float(sim) 
                })
                seen_glosas.add(glosa_palabra)

    # 6. Ordenar por similitud para que el LLM vea las mejores primero
    resultados.sort(key=lambda x: x['similitud_score'], reverse=True)
    
    # Devolvemos las mejores K que pasaron el umbral
    return resultados[:top_k]
# =================================================================
# BLOQUE 3 (MODIFICADO): LÓGICA DE TRADUCCIÓN CON LANGCHAIN
# =================================================================


# 1. Definir el modelo de lenguaje (LLM) de LangChain 
llm = ChatOpenAI(
    model=CONFIG['openai']['chat_model'],
    openai_api_base=OPENAI_BASE_URL,
    openai_api_key=GITHUB_TOKEN,
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 2. Definir la plantilla del prompt (¡MUY MODIFICADA!)
prompt_template = """
Eres un traductor experto de español a glosas de la Lengua de Señas Chilena (LSCh). Tu única tarea es analizar el texto de entrada y seleccionar el subconjunto MÁS apropiado de glosas candidatas, ordenándolas correctamente.

**Contexto de la conversación anterior (si es relevante):**
{contexto}

**Texto a traducir:**
{texto}

**Glosas Candidatas (con su descripción para ayudarte a elegir):**
{candidatas_contexto}

**Reglas OBLIGATORIAS E INQUEBRANTABLES:**
1.  **SELECCIONA DESDE LAS CANDIDATAS:** Usa ÚNICAMENTE glosas de la lista de candidatas (ej. si la candidata es {{"glosa": "CASA", ...}}, usa "CASA").
2.  **RELEVANCIA PRIMERO:** Elige solo las glosas que traduzcan DIRECTAMENTE el significado del texto. Usa las "descripcion" de las candidatas para evitar glosas que suenen parecido pero signifiquen algo distinto (ej. no elijas "NOCHE" si la candidata "BUENAS-NOCHES" es mejor).
3.  **NO REPITAS:** No uses la misma glosa más de una vez.
4.  **NO INVENTES:** No modifiques ni combines glosas.
5.  **ORDEN LÓGICO (¡MUY IMPORTANTE!):** Ordena las glosas finales para que sigan el orden lógico y gramatical del texto original en español, pero adaptado a la estructura de la LSCh (generalmente Sujeto-Objeto-Verbo, y adjetivos/tiempo después del sustantivo/verbo).
    * **Ejemplo de Orden:** "Yo voy a la casa roja"
    * Candidatas: [..., {{"glosa": "YO"}}, {{"glosa": "IR"}}, {{"glosa": "CASA"}}, {{"glosa": "ROJO"}}, ...]
    * **Salida Correcta:** {{"glosas": ["YO", "CASA", "ROJO", "IR"]}} (Sujeto, Objeto, Adjetivo, Verbo)
6.  **FORMATO JSON:** Devuelve la respuesta como un objeto JSON con una única clave "glosas" que contenga una lista de strings. Si ninguna glosa aplica, devuelve una lista vacía.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 3. Definir el parser para la salida JSON 
parser = JsonOutputParser()

# 4. Crear la "cadena" (chain) de procesamiento 
# Esta cadena ahora llama a la nueva función y formatea su salida
chain = (
    RunnablePassthrough.assign(
        # Llama a la nueva función y formatea su salida como un string JSON
        candidatas_contexto=lambda x: json.dumps(
            buscar_candidatas_contextuales(x["texto"]), 
            ensure_ascii=False,
            indent=2
        )
    )
    | prompt
    | llm
    | parser
)

# 5. Función 'process_text' 
# La única modificación es asegurarnos de que el LLM reciba el texto
# correcto (texto_marcado) y la validación final se mantenga.
def process_text():
    """Procesa el texto usando la cadena de LangChain y manejando números dígito por dígito."""
    global sistema_activo
    conversational_history = deque(maxlen=4) # Historial de contexto

    while sistema_activo:
        try:
            texto_original = text_queue.get(timeout=1)
            
            # === INICIO DEL BLOQUE DE MANEJO DE NÚMEROS ===
            numeros_como_glosas = re.findall(r'\d', texto_original)
            texto_sin_numeros = re.sub(r'\d+', '', texto_original).strip()
            
            glosas_de_palabras = []
            
            # Solo llamar a la cadena de LangChain si queda texto por traducir.
            if texto_sin_numeros:
                texto_marcado = marcar_pronombres(texto_sin_numeros)
                contexto_str = " ".join(conversational_history)

                try:
                    # Invocamos la cadena con el texto marcado y el contexto
                    response_json = chain.invoke({
                        "texto": texto_marcado, # Usamos el texto con pronombres marcados
                        "contexto": contexto_str if contexto_str else "No hay contexto previo."
                    })
                    glosas_crudas = response_json.get("glosas", [])

                    if not isinstance(glosas_crudas, list):
                        logging.warning(f"LangChain no devolvió una lista en el JSON: {glosas_crudas}")
                        glosas_crudas = []
                    
                    # Validamos que las glosas del LLM estén en nuestro glosario oficial
                    # Esta validación PRESERVA EL ORDEN entregado por el LLM
                    glosas_validadas = []
                    seen = set()
                    for g in glosas_crudas:
                        if g and isinstance(g, str):
                            g_clean = g.upper().strip()
                            if g_clean in glosario and g_clean not in seen:
                                glosas_validadas.append(g_clean)
                                seen.add(g_clean)
                    
                    glosas_de_palabras = glosas_validadas

                except Exception as e:
                    logging.error(f"Error al invocar o procesar la cadena de LangChain: {e}")
                    glosas_de_palabras = []
            
            # 4. Combinar las glosas de palabras (ordenadas por LLM) con los dígitos.
            glosas = glosas_de_palabras + numeros_como_glosas
            
            # === FIN DEL BLOQUE ===

            conversational_history.append(texto_original)
            resultado = {"texto": texto_original, "glosas": glosas}
            logging.info(f"Resultado: {resultado}")

            resultado_queue.put_nowait(resultado)

            # Guardar resultados
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
    pause_threshold = CONFIG['audio']['sentence_pause_ms'] / 1000

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
    print("Este es un módulo de transcripción. No está diseñado para ejecutarse directamente.")
    print("Por favor, ejecute fastapi_interface.py en su lugar.")