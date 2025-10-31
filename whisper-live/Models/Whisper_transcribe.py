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

# ==== Configuración de audio (SIN CAMBIOS) ====
samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

# ==== Colas (SIN CAMBIOS) ====
audio_queue = queue.Queue()
text_queue = queue.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo Whisper (SIN CAMBIOS) ====
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Detector de voz (SIN CAMBIOS) ====
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

# ==== Cliente OpenAI (Azure/GitHub) (SIN CAMBIOS) ====
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=os.environ.get("GITHUB_TOKEN")
)

# ==== Inicializar archivos (SIN CAMBIOS) ====
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Pronombres (SIN CAMBIOS) ====
try:
    with open(PRONOMBRES_PATH, "r", encoding="utf-8") as f:
        PRONOMBRES_MAP = json.load(f)
except Exception:
    # Fallback pequeño si el JSON no está presente
    PRONOMBRES_MAP = {
        "yo": "YO", "tú": "TÚ", "usted": "USTED", "él": "ÉL", "ella": "ELLA",
        "nosotros": "NOSOTROS", "ustedes": "USTEDES", "ellos": "ELLOS", "ellas": "ELLAS"
    }
    print("⚠️ Pronombres JSON no encontrado — usando mapa por defecto.")

# ==== Funciones de marcadores (SIN CAMBIOS) ====
def marcar_pronombres(texto):
    """Inserta marcadores [PRON:...] en el texto para la instrucción al traductor."""
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
    """Quita cualquier marcador tipo [PRON:...] para usar en la búsqueda FAISS."""
    return re.sub(r"\[PRON:[^\]]+\]\s*", "", texto).strip()

# =================================================================
# ⭐ BLOQUE 1: CARGA Y CREACIÓN DE FAISS (CON TEXTO CONTEXTUAL Y OPENAI)
# =================================================================
print("🧠 Cargando glosario y FAISS...")
glosario_df = pd.read_csv(GLOSARIO_PATH)

glosario_df = glosario_df.fillna('')

# 1. Crear la nueva columna de contexto a vectorizar (MISMO CAMBIO ANTERIOR)
glosario_df['Texto_Contexto'] = (
    glosario_df['Palabra'].str.upper().str.strip() + ". " +
    "Descripción: " + glosario_df['Descripción'] + ". " +
    "Categoría: " + glosario_df['Categoría'] + ". " +
    "Sinónimos: " + glosario_df['Sinónimos'] + ". " +
    "Antónimos: " + glosario_df['Antónimos']
)

# La lista de glosas para usar como labels (la "Palabra" original)
glosario = glosario_df["Palabra"].str.upper().str.strip().tolist()
# La lista de textos a vectorizar (el contexto combinado)
glosario_textos = glosario_df["Texto_Contexto"].tolist()

# Si no existe el índice FAISS, lo creamos una sola vez
if not os.path.exists(FAISS_INDEX_PATH):
    print("⚙️ Creando índice FAISS (primera vez) con contexto usando OpenAI...")
    embeddings = []
    batch_size = 100 
    
    # 2. Iterar sobre la lista de textos con contexto (glosario_textos) y usar OpenAI
    for i in range(0, len(glosario_textos), batch_size):
        batch = glosario_textos[i:i+batch_size]
        # Usar client.embeddings.create
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    
    glosario_embeddings = np.array(embeddings).astype("float32")
    
    np.save(EMB_PATH, glosario_embeddings) 
    
    dim = glosario_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(glosario_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    # Si el índice existe, cargamos el índice FAISS y el array de embeddings
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)

print(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")

# =================================================================
# ⭐ BLOQUE 2: FUNCIÓN DE BÚSQUEDA (REVERTIDA Y RÁPIDA: Vectorización por frase completa)
# =================================================================
def buscar_glosas(texto, top_k=20, threshold=0.35):
    """
    Busca glosas relevantes vectorizando la frase completa (método más rápido)
    y usando la API de OpenAI.
    """
    texto_para_buscar = quitar_marcadores(texto).strip()
    if not texto_para_buscar:
        return []

    # Vectorizar la frase completa con OpenAI (UNA SOLA LLAMADA)
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small", 
            input=texto_para_buscar
        ).data[0].embedding
    except Exception as e:
        print(f"⚠️ Error generando embedding para la frase: {e}")
        # Si falla la API aquí, no hay búsqueda, retorna []
        return []

    # Convertir a formato FAISS (1, dim) y realizar la búsqueda
    emb = np.array([emb]).astype("float32")
    
    # Aumentar top_k para capturar más glosas contextuales
    D, I = index.search(emb, top_k) 

    # Normalización y filtrado por umbral
    max_d, min_d = float(np.max(D)), float(np.min(D))
    denom = (max_d - min_d + 1e-6)
    similitudes = 1 - ((D - min_d) / denom) 
    
    resultados_totales = set()
    for idx, sim in zip(I[0], similitudes[0]):
        # Se mantiene el umbral original de OpenAI (0.35)
        if sim >= threshold: 
            resultados_totales.add(glosario[idx])

    return list(resultados_totales)


# ==== Traducción con GPT y FAISS (SIN CAMBIOS) ====
def traducir_a_glosas(texto):
    texto = texto.strip()
    if not texto:
        return []

    candidatas = buscar_glosas(texto)
    if not candidatas:
        print(f"⚠️ Sin glosas candidatas válidas para: '{texto}'")
        return []

    texto_sin_marcadores = quitar_marcadores(texto)
    es_palabra_unica = len(texto_sin_marcadores.split()) == 1

    # 🔹 Si es una sola palabra, no usamos GPT — devolvemos la más cercana
    if es_palabra_unica:
        return [candidatas[0]]

    # 🔹 Prompt extremadamente restrictivo
    prompt = f"""
Traduce el siguiente texto al formato de glosas de la Lengua de Señas Chilena (LSCh).

Texto original (puede incluir pronombres marcados como [PRON:YO]):
{texto}

Solo puedes usar glosas de la siguiente lista:
{", ".join(candidatas)}

Reglas obligatorias:
- Usa ÚNICAMENTE las glosas de la lista proporcionada.
- NO inventes, modifiques ni combines glosas.
- Si ninguna aplica, devuelve [].
- Si el texto tiene varias palabras, usa el orden LSCh (tema/sujeto → verbo → objeto).
- Devuelve SOLO una lista JSON válida, sin texto adicional.
Ejemplo: ["YO", "TRABAJAR", "ESCUELA"]
"""

    try:
        response = client.chat.completions.create(
            # Se usa un modelo rápido
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "system",
                    "content": "Eres un traductor de texto a glosas de la Lengua de Señas Chilena (LSCh). Solo puedes usar glosas existentes del diccionario."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = getattr(response.choices[0].message, "content", None)
        if not content:
            print("⚠️ GPT devolvió contenido vacío.")
            return []

        try:
            salida = json.loads(content)
        except Exception:
            print(f"⚠️ Error al parsear JSON: {content}")
            return []

        if isinstance(salida, dict) and "glosas" in salida:
            glosas = salida["glosas"]
        elif isinstance(salida, list):
            glosas = salida
        else:
            glosas = []

        glosas_filtradas = [
            g.upper() for g in glosas if g and g.upper() in glosario
        ]

        if not glosas_filtradas:
            print(f"⚠️ Ninguna glosa válida para '{texto}' → {glosas}")
            return []

        seen = set()
        glosas_finales = [g for g in glosas_filtradas if not (g in seen or seen.add(g))]

        return glosas_finales

    except Exception as e:
        print(f"⚠️ Error en GPT: {e}")
        return []


# ==== Flujo de audio (SIN CAMBIOS) ====
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
        texto_marcado = marcar_pronombres(texto)
        glosas = traducir_a_glosas(texto_marcado)

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
            json.dump(data, f, ensure_ascii=False, indent=4)

# ==== Control (SIN CAMBIOS) ====
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