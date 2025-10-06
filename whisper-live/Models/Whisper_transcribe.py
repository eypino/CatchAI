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
import numpy as np

# ==== Configuración de audio ====
samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

# ==== Colas ====
audio_queue = queue.Queue()
text_queue = queue.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo Whisper ====
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Detector de voz ====
vad = webrtcvad.Vad(2)

# ==== Rutas ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")

GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_glosas.csv")
TRANSCRIPCION_TXT = os.path.join(STORAGE_DIR, "transcripciones.txt")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")

# ==== Cliente OpenAI (Azure/GitHub) ====
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=os.environ.get("GITHUB_TOKEN")
)

# ==== Inicializar archivos ====
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Glosario y FAISS ====
print("🧠 Cargando glosario y FAISS...")
glosario_df = pd.read_csv(GLOSARIO_PATH)
glosario = glosario_df["Palabra"].str.upper().str.strip().tolist()

# Si no existe el índice FAISS, lo creamos una sola vez
if not os.path.exists(FAISS_INDEX_PATH):
    print("⚙️ Creando índice FAISS (primera vez)...")
    embeddings = [
        client.embeddings.create(model="text-embedding-3-small", input=g).data[0].embedding
        for g in glosario
    ]
    np.save(EMB_PATH, np.array(embeddings).astype("float32"))
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)

print(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")

# ==== Buscar glosas con FAISS ====
def buscar_glosas(texto, top_k=20):
    emb = client.embeddings.create(model="text-embedding-3-small", input=texto).data[0].embedding
    emb = np.array([emb]).astype("float32")
    D, I = index.search(emb, top_k)
    return [glosario[i] for i in I[0]]

# ==== Traducción con GPT y FAISS ====
def traducir_a_glosas(texto):
    candidatas = buscar_glosas(texto, top_k=25)

    prompt = f"""
Convierte la siguiente oración a glosas LSCh.
Usa SOLO glosas de la lista candidata.

Oración:
{texto}

Glosas candidatas:
{", ".join(candidatas)}

Reglas:
- Usa la menor cantidad de glosas posible para captar la idea principal.
- Nunca devuelvas deletreo.
- Devuelve SOLO una lista JSON de glosas.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un traductor de texto a glosas LSCh."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        salida = json.loads(response.choices[0].message.content)
        if isinstance(salida, dict) and "glosas" in salida:
            return salida["glosas"]
        elif isinstance(salida, list):
            return salida
        else:
            return []
    except Exception as e:
        print(f"⚠️ Error en GPT: {e}")
        return []

# ==== Flujo de audio ====
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
            data = json.load(f)
            data.append(resultado)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)

# ==== Control ====
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
