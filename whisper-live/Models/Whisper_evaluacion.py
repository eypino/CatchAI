import time
import numpy as np
from faster_whisper import WhisperModel
from jiwer import wer
import pandas as pd
import unicodedata
import re

# ==== CONFIGURACIONES A PROBAR ====
configs = [
    {"model_size": "small", "device": "cpu", "compute_type": "int8", "beam_size": 1},
    {"model_size": "medium", "device": "cpu", "compute_type": "int8", "beam_size": 1},
    {"model_size": "large-v3", "device": "cpu", "compute_type": "int8", "beam_size": 5},
]

# ==== AUDIO DE PRUEBA ====
audio_file = "hola_como_estas.m4a"  
reference_text = "hola como estas tu"  # la transcripción esperada

# ==== FUNCIONES AUXILIARES ====
def normalize(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-zñ\s]', '', text)  # quitar puntuación
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==== FUNCIÓN DE BENCHMARK ====
def benchmark(config, audio_file, reference_text):
    print(f"\n=== Probando {config} ===")
    model = WhisperModel(
        config["model_size"],
        device=config["device"],
        compute_type=config["compute_type"]
    )

    start = time.time()
    segments, info = model.transcribe(
        audio_file,
        beam_size=config["beam_size"],
        language="es"
    )
    end = time.time()

    # Reconstruir hipótesis de transcripción
    hypothesis = " ".join([seg.text.strip() for seg in segments])

    # Normalizar texto antes de WER
    ref_norm = normalize(reference_text)
    hyp_norm = normalize(hypothesis)
    error = wer(ref_norm, hyp_norm)

    # Métricas
    elapsed = end - start
    audio_duration = info.duration
    rtf = elapsed / audio_duration

    result = {
        "model": config["model_size"],
        "device": config["device"],
        "beam": config["beam_size"],
        "elapsed_s": round(elapsed, 2),
        "audio_s": round(audio_duration, 2),
        "RTF": round(rtf, 2),
        "WER": round(error, 3),
        "hypothesis": hypothesis
    }
    return result

# ==== CORRER TODOS LOS EXPERIMENTOS ====
results = []
for cfg in configs:
    results.append(benchmark(cfg, audio_file, reference_text))

# ==== MOSTRAR RESULTADOS ====
df = pd.DataFrame(results)
print(df)

# Exportar a CSV
df.to_csv("benchmark_results.csv", index=False)
