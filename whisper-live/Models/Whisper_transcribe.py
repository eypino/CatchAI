# whisper_procesamiento_transcribe.py
import sounddevice as sd
import numpy as np
import queue
import os
import threading
import webrtcvad
import string
import pandas as pd
from faster_whisper import WhisperModel

# ==== Configuración de audio ====
samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

audio_queue = queue.Queue()
text_queue = queue.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo Whisper ====
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Detector de voz ====
vad = webrtcvad.Vad(2)

# ==== Cargar glosario ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas relativas a Storage
GLOSARIO_PATH = os.path.join(BASE_DIR, "Storage", "diccionario_lsch_glosas_corregido.csv")
TRANSCRIPCION_PATH = os.path.join(BASE_DIR, "Storage", "transcripciones.txt")
try:
    glosario = set(pd.read_csv(GLOSARIO_PATH)["Palabra"].str.upper().str.strip().tolist())
except FileNotFoundError:
    glosario = set()

# ==== Variables compartidas ====
resultados_globales = []

# ==== Funciones principales ====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frame_per_block):
        print("🎙️ Grabando... (Ctrl+C para detener)")
        while True:
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

def transcribe_audio():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer = np.vstack((audio_buffer, block))

        while len(audio_buffer) >= frame_per_chunk:
            audio_data = audio_buffer[:frame_per_chunk].flatten().astype(np.float32)
            audio_buffer = audio_buffer[frame_per_chunk:]

            if not is_speech(audio_data):
                continue

            segments, info = model.transcribe(audio_data, beam_size=1, language="es")
            for segment in segments:
                if segment.no_speech_prob < 0.6 and segment.text.strip():
                    text_queue.put((segment.start, segment.end, segment.text))

def process_text():
    global resultados_globales
    while True:
        start_time, end_time, texto = text_queue.get()
        glosas = traducir_a_glosas(texto)
        resultado = {
            "inicio": round(start_time, 2),
            "fin": round(end_time, 2),
            "texto": texto,
            "glosas": glosas
        }
        resultados_globales.append(resultado)
        print(resultado)

def iniciar_sistema():
    """Inicia los hilos de grabación, transcripción y procesamiento"""
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_text, daemon=True).start()
    threading.Thread(target=transcribe_audio, daemon=True).start()
    print("🚀 Sistema de transcripción en vivo iniciado")
