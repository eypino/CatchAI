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
import re
from sentence_transformers import SentenceTransformer 
import ollama 
from ollama import Client as OllamaClient

# ==== Configuración y setup (SIN CAMBIOS) ====
samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1
frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)
audio_queue = queue.Queue()
text_queue = queue.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
vad = webrtcvad.Vad(2)

# ==== Rutas (SIN CAMBIOS) ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")
GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_definitivo.csv")
TRANSCRIPCION_TXT = os.path.join(STORAGE_DIR, "transcripciones.txt")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")
PRONOMBRES_PATH = os.path.join(STORAGE_DIR, "Pronombres_map.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")

# ==== Cliente Ollama para LLM y Embeddings ====
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL_NAME = "phi3:instruct" 
OLLAMA_EMBEDDING_MODEL = 'BAAI/bge-m3'

try:
    ollama_llm_client = OllamaClient(host=OLLAMA_BASE_URL)
    print(f"✅ Cliente Ollama para LLM conectado en {OLLAMA_BASE_URL}")
except Exception as e:
    print(f"❌ Error al conectar con Ollama LLM: {e}")

try:
    print(f"🧠 Cargando modelo de Embedding local: {OLLAMA_EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(OLLAMA_EMBEDDING_MODEL)
    print("✅ Modelo BGE-M3 cargado con éxito.")
except Exception as e:
    print(f"❌ Error al cargar SentenceTransformer: {e}")
    
# ==== Inicialización de archivos / Cargar Pronombres / Marcadores (SIN CAMBIOS) ====
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

try:
    with open(PRONOMBRES_PATH, "r", encoding="utf-8") as f:
        PRONOMBRES_MAP = json.load(f)
except Exception:
    PRONOMBRES_MAP = {
        "yo": "YO", "tú": "TÚ", "usted": "USTED", "él": "ÉL", "ella": "ELLA",
        "nosotros": "NOSOTROS", "ustedes": "USTEDES", "ellos": "ELLOS", "ellas": "ELLAS"
    }
    print("⚠️ Pronombres JSON no encontrado — usando mapa por defecto.")

def marcar_pronombres(texto):
    palabras = texto.lower().split()
    marcadas = []
    for p in palabras:
        p_clean = p.translate(str.maketrans('', '', string.punctuation))
        if p_clean in PRONOMBRES_MAP:
            marcadas.append(PRONOMBRES_MAP[p_clean])
        else:
            marcadas.append(p)
    return " ".join(marcadas)

def quitar_marcadores(texto):
    return re.sub(r"[^\w\s]", "", texto).strip()

# ⭐ FUNCIÓN DE LECTURA DE HISTORIAL
def get_historial_contexto(max_segments=5):
    """
    Lee los últimos N segmentos de la transcripción para dar contexto al LLM.
    """
    try:
        with open(TRANSCRIPCION_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    
    contexto = []
    # Tomar solo los últimos 5 segmentos
    for item in data[-max_segments:]:
        # Formato el historial para que sea legible por el LLM
        # Excluimos las glosas del historial para no sobrecargar el prompt
        contexto.append(
            f"Texto: {item['texto']}"
        )
    
    return "\n".join(contexto)


# =================================================================
# BLOQUE 1: CREACIÓN DE FAISS (BGE-M3)
# =================================================================
print("🧠 Cargando glosario y FAISS...")
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
    print(f"⚙️ Creando índice FAISS (primera vez) con {OLLAMA_EMBEDDING_MODEL}...")
    glosario_embeddings = embedding_model.encode(
        glosario_textos, 
        convert_to_numpy=True, 
        show_progress_bar=True
    ).astype("float32")
    np.save(EMB_PATH, glosario_embeddings) 
    dim = glosario_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(glosario_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)
print(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")

# =================================================================
# BLOQUE 2: FUNCIÓN DE BÚSQUEDA (BGE-M3)
# =================================================================
def buscar_glosas(texto, top_k=50, threshold=0.05):
    texto_para_buscar = texto.strip()
    if not texto_para_buscar:
        return {}
    try:
        emb_array = embedding_model.encode(texto_para_buscar)
    except Exception as e:
        print(f"⚠️ Error generando embedding para la frase: {e}")
        return {}
    emb = np.array([emb_array]).astype("float32")
    D, I = index.search(emb, top_k) 
    
    max_d, min_d = float(np.max(D)), float(np.min(D))
    denom = (max_d - min_d + 1e-6)
    similitudes = 1 - ((D - min_d) / denom) 
    
    resultados_similitud = {}
    for idx, sim in zip(I[0], similitudes[0]):
        if sim >= threshold: 
            resultados_similitud[glosario[idx]] = sim

    print(f"DEBUG (FAISS): Texto '{texto.strip()}' -> Encontradas {len(resultados_similitud)} candidatas (Umbral {threshold})")
    
    return resultados_similitud

# =================================================================
# ⭐ FUNCIÓN DE TRADUCCIÓN CON PHI-3 INSTRUCT (CON HISTORIAL)
# =================================================================
def traducir_a_glosas(texto):
    texto = texto.strip()
    if not texto:
        return []

    candidatas_map = buscar_glosas(texto, top_k=50, threshold=0.05)
    candidatas_glosas = list(candidatas_map.keys())

    # --- LÓGICA DE FALLBACK/COINCIDENCIA EXACTA ---
    if not candidatas_glosas:
        palabra_glosa = texto.upper().strip().translate(str.maketrans('', '', string.punctuation))
        if palabra_glosa in glosario:
             print(f"DEBUG (FALLBACK): Glosa exacta '{palabra_glosa}' aplicada.")
             return [palabra_glosa]
        
        print(f"⚠️ Sin glosas candidatas válidas para: '{texto}' (FAISS y Fallback fallaron)")
        return []
    # ---------------------------------------------

    # ⭐ OBTENER HISTORIAL DE CONTEXTO
    historial = get_historial_contexto(max_segments=5)


    # 🔹 Prompt para PHI-3 (AÑADIENDO HISTORIAL Y REGLAS)
    prompt = f"""
ERES UN TRADUCTOR AUTOMATIZADO A GLOSAS DE LENGUA DE SEÑAS CHILENA (LSCh). TU ÚNICO TRABAJO ES PRODUCIR UNA LISTA JSON PERFECTA.

---
HISTORIAL DE CONVERSACIÓN PREVIA:
{historial if historial else "No hay historial previo."}
---

Instrucción: Traduce el siguiente texto, usando el historial para desambiguar palabras.

Texto a traducir:
{marcar_pronombres(texto)} 

Glosas candidatas (SOLO PUEDES USAR ESTAS):
{", ".join(candidatas_glosas)}

REGLAS CRÍTICAS DE SALIDA:
1. La traducción debe seguir el orden gramatical LSCh (Tópico-Comentario o SVO).
2. Usa SOLO las glosas que estén estrictamente en la lista candidata.
3. Devuelve *ÚNICAMENTE* la lista JSON de glosas, sin ninguna clave envolvente (NO USES "glosas:", "LSCh:", etc.).
4. PRIORIZA glosas que son sustantivos, verbos o adjetivos directamente relacionados con el significado de la frase. IGNORA glosas irrelevantes (ej., 'PAN', 'CERA').
5. Si fallas en devolver el formato ["GLOSA", "GLOSA"], la operación falla.

Ejemplo de SALIDA ÚNICA y CORRECTA:
["HOLA", "BUENO", "SISTEMA"]
"""

    try:
        response = ollama_llm_client.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un traductor estricto de texto a glosas LSCh y solo devuelves JSON."},
                {"role": "user", "content": prompt}
            ],
            options={
                'temperature': 0.1,
                'num_predict': 50,
                'num_ctx': 2048
            },
            format='json'
        )

        content = response['message']['content']
        print(f"DEBUG (LLM): Raw LLM Output: {content}") 
        
        # 🔒 Lógica de Extracción Robusta (Solución al error del LLM)
        glosas = []
        try:
            salida = json.loads(content)
        except Exception:
            print(f"⚠️ Error al parsear JSON de Ollama. El LLM no devolvió JSON válido.")
            return []

        if isinstance(salida, list):
            glosas = salida
        elif isinstance(salida, dict):
            # Prioridad 1: Buscar las claves de lista esperadas
            if "glosas" in salida:
                glosas = salida["glosas"]
            elif "LSCh" in salida:
                glosas = salida["LSCh"]
            elif "translation" in salida:
                glosas = salida["translation"]
            elif "glosa" in salida:
                 glosas = salida["glosa"]
            
            # FALLBACK CRÍTICO: Si no se encontró ninguna lista, extraer las CLAVES
            if not glosas:
                print("DEBUG (LLM EXTRACT): Extrayendo glosas de las claves del diccionario (FALLBACK).")
                glosas = list(salida.keys())
            
        # Filtrado Final
        glosas_filtradas = [
            g.upper() for g in glosas if g and g.upper() in glosario
        ]

        if not glosas_filtradas:
            print(f"⚠️ LLM no devolvió glosas válidas del set candidato.")
            return []
        
        seen = set()
        glosas_finales = [g for g in glosas_filtradas if not (g in seen or seen.add(g))]

        return glosas_finales

    except Exception as e:
        print(f"⚠️ Error en LLM local (Ollama): {e}")
        return []

# ==== Flujo de audio, process_text, etc. (SIN CAMBIOS) ====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    global sistema_activo
    with sd.InputStream(samplerate=samplerate, channels=channels,
                         callback=audio_callback, blocksize=frame_per_block):
        print("🎙️ Grabando... Ctrl+C para detener.")
        while sistema_activo:
            sd.sleep(1000)

def is_speech(audio_data):
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    n_frames = len(audio_int16) // frame_size
    if n_frames == 0:
        return False

    speech_frames = 0
    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio_int16[start:end].tobytes()
        if vad.is_speech(frame, samplerate):
            speech_frames += 1

    return (speech_frames / n_frames) > 0.3

def transcribe_audio():
    global audio_buffer, sistema_activo
    while sistema_activo:
        if audio_queue.empty():
            continue
        block = audio_queue.get()
        audio_buffer = np.vstack((audio_buffer, block))

        while len(audio_buffer) >= frame_per_chunk and sistema_activo:
            audio_data = audio_buffer[:frame_per_chunk].flatten().astype(np.float32)
            audio_buffer = audio_buffer[frame_per_chunk:]

            if not is_speech(audio_data):
                continue

            segments, info = model.transcribe(audio_data, beam_size=1, language="es")
            for segment in segments:
                if segment.no_speech_prob < 0.6 and segment.text.strip():
                    text_queue.put((segment.start, segment.end, segment.text))

resultados_globales = []

def process_text():
    global resultados_globales, sistema_activo
    while sistema_activo:
        start_time, end_time, texto = text_queue.get()
        glosas = traducir_a_glosas(texto)

        resultado = {
            "inicio": round(start_time, 2),
            "fin": round(end_time, 2),
            "texto": texto,
            "glosas": glosas
        }

        resultados_globales.append(resultado)
        with open(TRANSCRIPCION_TXT, "a", encoding="utf-8") as f:
            f.write(str(resultado) + "\n")

        with open(TRANSCRIPCION_JSON, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                # Si el archivo está vacío o corrupto, lo inicializamos como lista vacía
                data = [] 
            data.append(resultado)
            f.seek(0)
            f.truncate() # Asegura que no queden restos del archivo anterior
            json.dump(data, f, ensure_ascii=False, indent=4)

def iniciar_sistema():
    global sistema_activo
    sistema_activo = True
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_text, daemon=True).start()
    threading.Thread(target=transcribe_audio, daemon=True).start()
    print("🚀 Sistema de transcripción + glosas LSCh iniciado")

def detener_sistema():
    global sistema_activo
    sistema_activo = False
    print("🛑 Sistema detenido")

if __name__ == "__main__":
    iniciar_sistema()
    while sistema_activo:
        sd.sleep(1000)