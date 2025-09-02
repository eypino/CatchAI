import sounddevice as sd
import numpy as np
import pandas as pd
import queue
import threading
import webrtcvad
from faster_whisper import WhisperModel
import string

# ==== Configuración de audio ====
samplerate = 16000 # Hertz
block_duration = 0.5 # segundos por bloque de lectura
chunk_duration = 2 # segundos por chunk a procesar
channels = 1 # mono

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

# ==== Colas para la comunicación entre hilos ====
audio_queue = queue.Queue() # Cola para pasar audio del micrófono a la transcripción
text_queue = queue.Queue() # Cola para pasar texto de la transcripción al procesamiento de glosas

audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo Whisper ====
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Detector de voz (VAD) ====
vad = webrtcvad.Vad(2) # 0=menos sensible, 3=muy sensible

# ==== Archivo y Glosario ====
transcription_file = "transcripciones.txt"
try:
    # Carga el glosario y elimina espacios en blanco para asegurar la limpieza
    glosario = set(pd.read_csv(r"C:\Users\TPROY_NBXX\Desktop\CatchAI\whisper-live\diccionario_lsch_glosas_corregido.csv")["Palabra"].str.upper().str.strip().tolist())
    print("✅ Glosario cargado correctamente.")
except FileNotFoundError:
    print("⚠️ Error: No se encontró el archivo del glosario. La traducción a glosas no funcionará.")
    glosario = set() # Usar un glosario vacío para evitar errores

# ==================================
# ==== HILO 1: CAPTURA DE AUDIO ====
# ==================================
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    """Captura audio del micrófono y lo pone en la audio_queue."""
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frame_per_block):
        print("🎙️ Grabando... Presiona Ctrl+C para detener.")
        while True:
            sd.sleep(1000)

def is_speech(audio_data):
    frame_duration = 30 # ms
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

# =========================================
# ==== HILO 2: TRANSCRIPCIÓN DE AUDIO ====
# =========================================
def transcribe_audio():
    """
    Saca audio de la audio_queue, lo transcribe con Whisper
    y pone el texto resultante en la text_queue.
    Esta función debe ser RÁPIDA.
    """
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
                    # Poner el texto transcrito en la cola para el siguiente hilo
                    # El formato es una tupla: (inicio, fin, texto)
                    text_queue.put((segment.start, segment.end, segment.text))

# ===============================================
# ==== HILO 3: PROCESAMIENTO DE TEXTO Y GLOSAS ====
# ===============================================
def deletrear_palabra(palabra):
    return list(palabra.upper())

def traducir_a_glosas(texto):
    # Limpia el texto de la puntuación antes de procesar
    texto_limpio = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto_limpio.upper().split()
    glosas = []
    for palabra in palabras:
        if palabra in glosario:
            glosas.append(palabra)
        else:
            glosas.extend(deletrear_palabra(palabra))
    return glosas

def process_text():
    """
    Saca texto de la text_queue, lo procesa para convertirlo a glosas
    y lo imprime/guarda. Esta función puede ser LENTA.
    """
    while True:
        # Espera hasta que haya un texto transcrito disponible
        start_time, end_time, texto = text_queue.get()

        # Realiza la tarea lenta de procesamiento de texto
        glosas = traducir_a_glosas(texto)

        # Formatea y muestra el resultado final
        line = f"[{start_time:.2f}s -> {end_time:.2f}s] {texto} -> {glosas}"
        print(line)

        with open(transcription_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ==== INICIAR LOS TRES HILOS ====
if __name__ == "__main__":
    # Hilo 1: Captura de audio (Productor de audio)
    threading.Thread(target=record_audio, daemon=True).start()
    # Hilo 3: Procesamiento de texto (Consumidor de texto)
    threading.Thread(target=process_text, daemon=True).start()
    # Hilo 2 (Principal): Transcripción (Consumidor de audio, Productor de texto)
    print("🚀 Sistema iniciado. Los tres hilos se ejecutan en paralelo.")
    transcribe_audio()