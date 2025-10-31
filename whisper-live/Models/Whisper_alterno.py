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
# ❌ Quitamos el cliente de Ollama para el LLM, ya no es necesario.

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
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario_O.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings_O.npy")

# ==== LLM Cliente Eliminado / Reemplazado por Ranking Semántico ====
OLLAMA_EMBEDDING_MODEL = 'BAAI/bge-m3'

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

# ⭐ Función de lectura de historial (Ya no es necesaria sin LLM, pero se mantiene si se quiere usar el contexto en el ranking)
def get_historial_contexto(max_segments=5):
    try:
        with open(TRANSCRIPCION_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    
    contexto = []
    for item in data[-max_segments:]:
        contexto.append(
            f"Texto: {item['texto']}"
        )
    
    return "\n".join(contexto)


# =================================================================
# BLOQUE 1: CREACIÓN DE FAISS (BGE-M3) - SIN CAMBIOS
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
# BLOQUE 2: FUNCIÓN DE BÚSQUEDA (BGE-M3) - SIN CAMBIOS
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
            # Guarda la glosa y su score de similitud
            resultados_similitud[glosario[idx]] = sim

    print(f"DEBUG (FAISS): Texto '{texto.strip()}' -> Encontradas {len(resultados_similitud)} candidatas (Umbral {threshold})")
    
    return resultados_similitud

# =================================================================
# ⭐ FUNCIÓN DE TRADUCCIÓN BASADA EN RANKING (SIN LLM)
# =================================================================
def traducir_a_glosas(texto):
    texto = texto.strip()
    if not texto:
        return []

    # Obtener el mapeo de glosa:similitud (ranking)
    candidatas_map = buscar_glosas(texto, top_k=50, threshold=0.05)
    
    if not candidatas_map:
        # Fallback de coincidencia exacta (mantenido)
        palabra_glosa = texto.upper().strip().translate(str.maketrans('', '', string.punctuation))
        if palabra_glosa in glosario:
             return [palabra_glosa]
        
        return []
    
    # 1. Obtener las palabras del texto (incluyendo pronombres marcados)
    palabras_texto_glosa = marcar_pronombres(texto).upper().split()
    
    # 2. Ordenar las candidatas por score descendente
    # Selecciona solo las 10 mejores por score (para evitar basura)
    mejores_candidatas = sorted(
        candidatas_map.items(), key=lambda item: item[1], reverse=True
    )[:10] 
    
    # 3. Lógica de selección e intento de orden
    glosas_finales = []
    
    # Intentamos mantener el orden del texto original (palabras_texto_glosa) 
    # y sustituirlas por la glosa de mayor ranking que coincida.
    
    candidatas_usadas = set() # Para evitar duplicados en el output
    
    # Intenta mapear palabra del texto original -> Glosa de alto ranking
    for palabra in palabras_texto_glosa:
        glosa_encontrada = None
        
        # A) Coincidencia directa (Pronombres o Glosa exacta)
        if palabra in PRONOMBRES_MAP.values() or palabra in glosario:
            glosa_encontrada = palabra
        
        # B) Coincidencia semántica: Mapear a la glosa de mayor ranking que tenga la palabra
        if not glosa_encontrada:
            # Busca la glosa con el mejor score que contenga o se relacione con la palabra
            for glosa_candidata, score in mejores_candidatas:
                if palabra in glosa_candidata.split() or glosa_candidata == palabra:
                    glosa_encontrada = glosa_candidata
                    break # Tomar la primera (mejor score) que coincida
        
        if glosa_encontrada and glosa_encontrada not in candidatas_usadas:
            glosas_finales.append(glosa_encontrada)
            candidatas_usadas.add(glosa_encontrada)
            
    # 4. Fallback final: Si el mapeo palabra por palabra falló, solo toma las 3 mejores glosas del ranking.
    if not glosas_finales and mejores_candidatas:
        glosas_finales = [g for g, s in mejores_candidatas[:3]]

    # Limpieza final de duplicados (aunque el proceso anterior debería ser limpio)
    seen = set()
    return [g for g in glosas_finales if not (g in seen or seen.add(g))]


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
                data = [] 
            data.append(resultado)
            f.seek(0)
            f.truncate()
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