import sounddevice as sd
import numpy as np
import queue
import threading
import webrtcvad
from faster_whisper import WhisperModel

# ==== Configuración de audio ====
samplerate = 16000  # Hertz
block_duration = 0.5  # segundos por bloque de lectura (chunks más pequeños)
chunk_duration = 2    # segundos por chunk a procesar
channels = 1          # mono

frame_per_chunk = int(samplerate * chunk_duration)
frame_per_block = int(samplerate * block_duration)

audio_queue = queue.Queue()
audio_buffer = np.zeros((0, channels), dtype=np.float32)

# ==== Modelo ====
model_size = "small" 
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Detector de voz (VAD) ====
vad = webrtcvad.Vad(2)  # 0=menos sensible, 3=muy sensible

# ==== Archivo de transcripción ====
transcription_file = "transcripciones.txt"

# ==== Funciones ====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=samplerate, channels=channels, 
                        callback=audio_callback, blocksize=frame_per_block):
        print("🎙 Grabando... Presiona Ctrl+C para detener.")
        while True:
            sd.sleep(1000)

def is_speech(audio_data):
    frame_duration = 30  # ms
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
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer = np.vstack((audio_buffer, block))

        # Procesar en chunks cortos para baja latencia
        while len(audio_buffer) >= frame_per_chunk:
            audio_data = audio_buffer[:frame_per_chunk].flatten().astype(np.float32)
            audio_buffer = audio_buffer[frame_per_chunk:]  # conservar sobrante

            if not is_speech(audio_data):
                continue

            segments, info = model.transcribe(audio_data, beam_size=1, language="es")

            for segment in segments:
                if segment.no_speech_prob < 0.6:
                    line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                    print(line)
                    with open(transcription_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

# ==== Iniciar grabación y transcripción ====
threading.Thread(target=record_audio, daemon=True).start()
transcribe_audio()
