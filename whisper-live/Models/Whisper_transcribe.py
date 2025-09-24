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

# ==== Rutas de archivos ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")
# Archivos dentro de Storage
GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_glosas.csv")
TRANSCRIPCION_TXT = os.path.join(STORAGE_DIR, "transcripciones.txt")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")

# Inicializar archivo JSON vacío si no existe
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Glosario ====
try:
    glosario = set(pd.read_csv(GLOSARIO_PATH)["Palabra"].str.upper().str.strip().tolist())
    print(f"✅ Glosario cargado desde {GLOSARIO_PATH}")
except FileNotFoundError:
    print(f"⚠️ No se encontró el archivo en {GLOSARIO_PATH}")
    glosario = set()

# ==== Estado del sistema ====
sistema_activo = True   # 👈 Flag de control

# ==== Funciones ====
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

def deletrear_palabra(palabra):
    return list(palabra.upper())

def traducir_a_glosas(texto):
    texto_limpio = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto_limpio.upper().split()
    glosas = []
    for palabra in palabras:
        if palabra in glosario:
            glosas.append(palabra)
        else:
            glosas.extend(deletrear_palabra(palabra))
    return glosas

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

        resultados_globales.append(resultado)   # 👈 ahora FastAPI podrá leerlo
        
        # Guardar en TXT
        with open(TRANSCRIPCION_TXT, "a", encoding="utf-8") as f:
            f.write(str(resultado) + "\n")

        # Guardar en JSON
        with open(TRANSCRIPCION_JSON, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(resultado)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)

# ==== Control del sistema ====
def iniciar_sistema():
    """Inicia los hilos de grabación, transcripción y procesamiento"""
    global sistema_activo
    sistema_activo = True
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_text, daemon=True).start()
    threading.Thread(target=transcribe_audio, daemon=True).start()
    print("🚀 Sistema de transcripción en vivo iniciado")

def detener_sistema():
    """Detiene la ejecución de los hilos"""
    global sistema_activo
    sistema_activo = False
    print("🛑 Sistema detenido")

# ==== Ejecución directa ====
if __name__ == "__main__":
    iniciar_sistema()
    while sistema_activo:
        sd.sleep(1000)
