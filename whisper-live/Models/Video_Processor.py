# Video_Processor.py

import numpy as np
import pandas as pd
import os
import json
import faiss
from openai import OpenAI
from faster_whisper import WhisperModel
# ¡NUEVA LIBRERÍA! Necesitarás instalarla: pip install moviepy
from moviepy.editor import VideoFileClip
from pydub import AudioSegment 

# ==== Configuración de audio y segmentación ====
# La tasa de muestreo debe ser 16000 para Whisper
samplerate = 16000
# Duración de cada segmento de audio para transcribir (e.g., 30 segundos)
# Un segmento más largo da más contexto para la traducción a glosas.
segment_duration_sec = 30 
# Factor para superponer segmentos, lo que ayuda a Whisper a no cortar palabras
# en los bordes. E.g., 2 segundos de superposición.
overlap_sec = 2 

# Duración en milisegundos para las librerías de audio
segment_duration_ms = segment_duration_sec * 1000
overlap_ms = overlap_sec * 1000

# ==== Modelo Whisper ====
# Se recomienda usar un modelo pequeño o medio para procesamiento por lotes
model_size = "medium" 
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ==== Rutas ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Asumiendo que SCRIPT_DIR está dentro de BASE_DIR
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")

GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_glosas.csv")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "video_transcripcion.json") # Archivo de salida
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")

# Asegurar que el directorio Storage exista
os.makedirs(STORAGE_DIR, exist_ok=True)

# ==== Cliente OpenAI (Azure/GitHub) ====
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=os.environ.get("GITHUB_TOKEN")
)

# ==== Inicializar archivo JSON de transcripción ====
# Se reinicia el archivo para cada procesamiento de video
with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Glosario y FAISS (Misma lógica) ====
print("🧠 Cargando glosario y FAISS...")
try:
    glosario_df = pd.read_csv(GLOSARIO_PATH)
    glosario = glosario_df["Palabra"].str.upper().str.strip().tolist()

    if not os.path.exists(FAISS_INDEX_PATH):
        print("⚙️ Creando índice FAISS (primera vez)...")
        # Generación de embeddings omitida por brevedad, asumiendo que ya existen
        # o que la lógica de tu script original ya lo manejará.
        # Si tienes problemas, revisa que GLOSARIO_PATH y EMB_PATH sean correctos.
        if not os.path.exists(EMB_PATH):
            print("❌ Error: Falta archivo de embeddings. Ejecuta el script original o genera los embeddings.")
            exit()
            
        embeddings = np.load(EMB_PATH)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
    
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")
except Exception as e:
    print(f"❌ Error al cargar glosario/FAISS: {e}")
    exit()

# ==== Buscar glosas con FAISS (Misma función) ====
def buscar_glosas(texto, top_k=25): # Aumentar un poco el top_k para más contexto
    emb = client.embeddings.create(model="text-embedding-3-small", input=texto).data[0].embedding
    emb = np.array([emb]).astype("float32")
    D, I = index.search(emb, top_k)
    return [glosario[i] for i in I[0]]

# ==== Traducción con GPT y FAISS (Misma función) ====
def traducir_a_glosas(texto):
    candidatas = buscar_glosas(texto, top_k=30) # Aumentar un poco el top_k para más contexto

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

# ==== Flujo principal de procesamiento de video ====

def extract_and_segment_audio(video_path):
    """Extrae el audio del video y lo divide en segmentos con superposición."""
    print(f"🎥 Procesando audio de: {video_path}")
    
    try:
        # 1. Extraer audio y convertir a AudioSegment
        clip = VideoFileClip(video_path)
        # Escribir el audio temporalmente para que pydub pueda cargarlo
        temp_audio_path = os.path.join(STORAGE_DIR, "temp_audio.mp3")
        clip.audio.write_audiofile(temp_audio_path, logger=None)
        
        # Cargar con pydub para manipulación fácil (usa FFmpeg internamente)
        audio = AudioSegment.from_file(temp_audio_path)
        audio = audio.set_frame_rate(samplerate).set_channels(1) # Asegurar 16kHz, mono
        
        os.remove(temp_audio_path)
        clip.close()

    except Exception as e:
        print(f"❌ Error al extraer audio del video: {e}")
        return []

    # 2. Segmentar el audio
    segments = []
    start_ms = 0
    total_duration_ms = len(audio)

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + segment_duration_ms, total_duration_ms)
        segment = audio[start_ms:end_ms]
        
        # Convertir el segmento a un array numpy de float32 normalizado
        # (similar a como lo espera Whisper)
        samples = np.array(segment.get_array_of_samples())
        samples = samples.astype(np.float32) / (1 << 15) # Normalizar de int16 a float32
        
        segments.append({
            "audio_data": samples,
            "start_time": start_ms / 1000.0, # En segundos
            "end_time": end_ms / 1000.0,     # En segundos
        })
        
        # Avanzar el inicio, aplicando la superposición
        start_ms += segment_duration_ms - overlap_ms

    print(f"✂️ Audio segmentado en {len(segments)} trozos.")
    return segments

def process_video_segments(video_path):
    """Procesa un video completo en lotes, transcribe y traduce a glosas."""
    
    segments = extract_and_segment_audio(video_path)
    if not segments:
        return
    
    resultados_globales = []
    
    for i, seg in enumerate(segments):
        print(f"\n🔬 Procesando segmento {i+1}/{len(segments)} ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)...")
        
        # 1. Transcripción con Faster Whisper
        # El modelo transcribe y segmenta internamente el trozo de 30s.
        segments_whisper, info = model.transcribe(
            seg["audio_data"], 
            beam_size=5, 
            language="es",
            # No. Aquí pasamos el audio sin segmentar para que Whisper decida la segmentación
        )
        
        # 2. Procesar las transcripciones internas de Whisper
        for whisper_seg in segments_whisper:
            texto = whisper_seg.text.strip()
            if whisper_seg.no_speech_prob < 0.6 and texto:
                # Calcular el tiempo global sumando el inicio del segmento grande
                start_time_global = seg['start_time'] + whisper_seg.start
                end_time_global = seg['start_time'] + whisper_seg.end
                
                print(f"   [T] ({start_time_global:.1f}s) {texto}")

                # 3. Traducción a Glosas con GPT
                glosas = traducir_a_glosas(texto)
                print(f"   [G] {glosas}")
                
                resultado = {
                    "inicio": round(start_time_global, 2),
                    "fin": round(end_time_global, 2),
                    "texto": texto,
                    "glosas": glosas
                }

                resultados_globales.append(resultado)
                
    # 4. Guardar resultados en el archivo JSON
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados_globales, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Proceso completado. Resultados guardados en {TRANSCRIPCION_JSON}")


if __name__ == "__main__":
    # --- DEBES CAMBIAR ESTA RUTA ---
    VIDEO_FILE_PATH = r"C:\Users\luism\Desktop\proyecto\CatchAI\whisper-live\Storage\T13_15segundos.mp4"
    # --- DEBES CAMBIAR ESTA RUTA ---
    
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"❌ Error: Archivo de video no encontrado en: {VIDEO_FILE_PATH}")
    else:
        process_video_segments(VIDEO_FILE_PATH)