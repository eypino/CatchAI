import sounddevice as sd
import numpy as np
import pandas as pd
import queue
import threading
from faster_whisper import WhisperModel
import string
import os
import json
import faiss
from openai import OpenAI
import re
import logging
import time
import asyncio
from collections import deque
from config_github_token import obtener_token
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# =================================================================
# BLOQUE 0: CONFIGURACIÓN Y LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)

def load_config():
    """Carga la configuración desde config.json o usa valores por defecto."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
            logging.info("Archivo config.json cargado exitosamente.")
            return json.load(f)
    except FileNotFoundError:
        logging.warning("No se encontró config.json. Usando config por defecto.")
        return {
          "audio": {"samplerate": 16000, "block_duration_ms": 1000, "chunk_duration_ms": 10000, "context_duration_ms": 2000, "channels": 1},
          "vad": {"threshold": 0.5, "min_silence_duration_ms": 1000},
          "whisper": {"model_size": "small", "device": "cpu", "compute_type": "int8", "beam_size": 1, "no_speech_prob": 0.4},
          "faiss": {"top_k": 100, "similarity_threshold": 0.10},
          "openai": {"embedding_model": "text-embedding-3-small", "chat_model": "gpt-4o-mini"}
        }

CONFIG = load_config()

# ==== Configuración de audio ====
samplerate = CONFIG['audio']['samplerate']
block_duration_ms = CONFIG['audio']['block_duration_ms']
channels = CONFIG['audio']['channels']
chunk_duration_ms = CONFIG['audio'].get('chunk_duration_ms', 10000)
context_duration_ms = CONFIG['audio'].get('context_duration_ms', 2000)
frame_per_chunk = int(samplerate * chunk_duration_ms / 1000)

# ==== Colas ====
audio_queue = queue.Queue()
# text_queue YA NO ES NECESARIA, EL NUEVO HILO HACE TODO
resultado_queue = asyncio.Queue() # La cola final para FastAPI
audio_buffer = np.zeros((0, channels), dtype=np.float32)
sistema_activo = False

# ==== Modelo Whisper ====
model_size = CONFIG['whisper']['model_size']
logging.info(f"Cargando modelo Whisper '{model_size}' en dispositivo '{CONFIG['whisper']['device']}'...")
model = WhisperModel(model_size, device=CONFIG['whisper']['device'], compute_type=CONFIG['whisper']['compute_type'])
logging.info("✅ Modelo Whisper cargado.")

# ==== Rutas ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")
GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_definitivo.csv")
TRANSCRIPCION_TXT = os.path.join(STORAGE_DIR, "transcripciones.txt")
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")
PRONOMBRES_PATH = os.path.join(STORAGE_DIR, "Pronombres_map.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")

# ==== Cliente OpenAI (Azure/GitHub) ====
GITHUB_TOKEN = obtener_token()
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=GITHUB_TOKEN)

# ==== Inicializar archivos ====
if not os.path.exists(TRANSCRIPCION_JSON):
    with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==== Cargar Pronombres ====
try:
    with open(PRONOMBRES_PATH, "r", encoding="utf-8") as f:
        PRONOMBRES_MAP = json.load(f)
except Exception:
    logging.warning("Pronombres JSON no encontrado — usando mapa por defecto.")
    PRONOMBRES_MAP = {
        "yo": "YO", "tú": "TÚ", "usted": "USTED", "él": "ÉL", "ella": "ELLA",
        "nosotros": "NOSOTROS", "ustedes": "USTEDES", "ellos": "ELLOS", "ellas": "ELLAS"
    }

# ==== NUEVO: Conversor de Números ====
NUMBER_WORDS_MAP = {
    "cero": "0", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
    "diez": "10"
}

def convertir_palabras_numeros(texto):
    palabras = texto.split()
    convertidas = []
    for p in palabras:
        p_clean = p.translate(str.maketrans('', '', string.punctuation)).lower()
        if p_clean in NUMBER_WORDS_MAP:
            convertidas.append(NUMBER_WORDS_MAP[p_clean])
        else:
            convertidas.append(p)
    return " ".join(convertidas)

# ==== Funciones de marcadores ====
def marcar_pronombres(texto):
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
    return re.sub(r"\[PRON:[^\]]+\]\s*", "", texto).strip()

# =================================================================
# BLOQUE 1: CARGA Y CREACIÓN DE FAISS (Sin cambios)
# =================================================================
logging.info("🧠 Cargando glosario y FAISS...")
glosario_df = pd.read_csv(GLOSARIO_PATH)
glosario_df = glosario_df.fillna('')
glosario_df['Texto_Contexto'] = (
    glosario_df['Palabra'].str.upper().str.strip() + ". " +
    "Descripción: " + glosario_df['Descripción'] + ". " +
    "Categoría: " + glosario_df['Categoría'] + ". " +
    "Sinónimos: " + glosario_df['Sinónimos'] + ". " +
    "Antónimos: " + glosario_df['Antónimos']
)
glosario = set(glosario_df["Palabra"].str.upper().str.strip().tolist())

if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(EMB_PATH):
    logging.info("⚙️ Creando índice FAISS (primera vez) con contexto usando OpenAI...")
    glosario_textos = glosario_df["Texto_Contexto"].tolist()
    embeddings = []
    batch_size = 100
    for i in range(0, len(glosario_textos), batch_size):
        batch = glosario_textos[i:i+batch_size]
        resp = client.embeddings.create(model=CONFIG['openai']['embedding_model'], input=batch)
        embeddings.extend([d.embedding for d in resp.data])
        time.sleep(0.5) 
    glosario_embeddings = np.array(embeddings).astype("float32")
    np.save(EMB_PATH, glosario_embeddings)
    dim = glosario_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(glosario_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    embeddings = np.load(EMB_PATH)
    dim = embeddings.shape[1]
    index = faiss.read_index(FAISS_INDEX_PATH)

logging.info(f"✅ Glosario y FAISS cargados ({len(glosario)} palabras).")

# =================================================================
# BLOQUE 2: FUNCIÓN DE BÚSQUEDA CONTEXTUAL (Sin cambios)
# =================================================================
def buscar_candidatas_contextuales(texto, top_k=CONFIG['faiss']['top_k'], threshold=CONFIG['faiss']['similarity_threshold']):
    texto_para_buscar = quitar_marcadores(texto).strip()
    if not texto_para_buscar:
        return []
    try:
        resp = client.embeddings.create(model=CONFIG['openai']['embedding_model'], input=[texto_para_buscar])
        embedding_query = np.array([resp.data[0].embedding]).astype("float32")
    except Exception as e:
        logging.error(f"Error generando embedding para la consulta: {e}")
        return []
    D, I = index.search(embedding_query, top_k)
    resultados = []
    indices = I[0]
    distancias = D[0]
    max_d, min_d = float(np.max(distancias)), float(np.min(distancias))
    denom = (max_d - min_d + 1e-6) 
    similitudes = 1 - ((distancias - min_d) / denom)
    seen_glosas = set()
    for idx, sim in zip(indices, similitudes):
        if sim >= threshold:
            glosa_palabra = glosario_df.iloc[idx]['Palabra'].upper().strip()
            if glosa_palabra not in seen_glosas:
                glosa_desc = glosario_df.iloc[idx]['Descripción']
                resultados.append({
                    "glosa": glosa_palabra,
                    "descripcion": glosa_desc,
                    "similitud_score": float(sim) 
                })
                seen_glosas.add(glosa_palabra)
    resultados.sort(key=lambda x: x['similitud_score'], reverse=True)
    return resultados[:top_k]

# =================================================================
# BLOQUE 3: LÓGICA DE TRADUCCIÓN CON LANGCHAIN
# =================================================================

# 1. Definir el modelo de lenguaje (LLM) de LangChain 
llm = ChatOpenAI(
    model=CONFIG['openai']['chat_model'],
    openai_api_base=OPENAI_BASE_URL,
    openai_api_key=GITHUB_TOKEN,
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 2. Definir la plantilla del prompt (V7 - MÁS ESTRICTA)
prompt_template = """
Eres un traductor experto de español a glosas de la Lengua de Señas Chilena (LSCh). Tu única tarea es analizar el texto de entrada y seleccionar el subconjunto MÁS apropiado de glosas candidatas, ordenándolas correctamente.

**Contexto de la conversación anterior (si es relevante):**
{contexto}

**Texto a traducir:**
{texto}

**Glosas Candidatas (con su descripción para ayudarte a elegir):**
{candidatas_contexto}

**Reglas OBLIGATORIAS E INQUEBRANTABLES:**
1.  **SELECCIONA DESDE LAS CANDIDATAS (¡SÉ ESTRICTO!):** Estás recibiendo MUCHAS candidatas de baja similitud (top_k=100, umbral=0.10). Tu trabajo es RECHAZARLAS si no traducen el significado. Usa ÚNICAMENTE glosas que sean la traducción correcta.
2.  **REGLA DE "FALSOS AMIGOS" (LA MÁS IMPORTANTE):** DEBES RECHAZAR glosas que sean semánticamente incorrectas, aunque suenen parecido. Si el texto dice "audio", RECHAZA "audífono". Si dice "personas", RECHAZA "muchas-veces". Es MEJOR omitir una glosa que mostrar una glosa INCORRECTA. Usa la "descripcion" de la candidata para confirmar el significado exacto.
3.  **NO SUSTITUIR:** Si la glosa exacta no está en las candidatas (ej. el texto dice "probando") pero una parecida sí (ej. "EXPERIMENTO"), RECHAZA "EXPERIMENTO". No sustituyas, solo traduce lo que esté presente.
4.  **NO REPITAS:** No uses la misma glosa más de una vez.
5.  **NO INVENTES:** No modifiques ni combines glosas.
6.  **ORDEN LÓGICO (¡MUY IMPORTANTE!):** Ordena las glosas finales para que sigan el orden lógico y gramatical del texto original en español, pero adaptado a la estructura de la LSCh (generalmente Sujeto-Objeto-Verbo).
    * **Ejemplo de Orden:** "Yo voy a la casa roja" -> {{"glosas": ["YO", "CASA", "ROJO", "IR"]}}
7.  **FORMATO JSON:** Devuelve la respuesta como un objeto JSON con una única clave "glosas" que contenga una lista de strings. Si ninguna glosa aplica, devuelve una lista vacía.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 3. Definir el parser para la salida JSON 
parser = JsonOutputParser()

# 4. Crear la "cadena" (chain) de procesamiento 
chain = (
    RunnablePassthrough.assign(
        candidatas_contexto=lambda x: json.dumps(
            buscar_candidatas_contextuales(x["texto"]), 
            ensure_ascii=False,
            indent=2
        )
    )
    | prompt
    | llm
    | parser
)

# 5. Función 'process_text' (AHORA ES LLAMADA INTERNAMENTE)
def process_text_segment(texto_original, conversational_history):
    """
    Procesa UN SOLO segmento de texto. Esta función es llamada por
    el hilo principal de la tubería.
    """
    
    # 1. Convertir palabras-número a dígitos (ej. "uno" -> "1")
    texto_con_numeros = convertir_palabras_numeros(texto_original)
    
    # 2. Extraer dígitos
    numeros_como_glosas = re.findall(r'\d', texto_con_numeros)
    texto_sin_numeros = re.sub(r'\d+', '', texto_con_numeros).strip()
    
    glosas_de_palabras = []
    
    # 3. Llamar a LangChain solo si queda texto
    if texto_sin_numeros:
        texto_marcado = marcar_pronombres(texto_sin_numeros)
        contexto_str = " ".join(conversational_history)

        try:
            response_json = chain.invoke({
                "texto": texto_marcado,
                "contexto": contexto_str if contexto_str else "No hay contexto previo."
            })
            glosas_crudas = response_json.get("glosas", [])

            if not isinstance(glosas_crudas, list):
                logging.warning(f"LangChain no devolvió una lista en el JSON: {glosas_crudas}")
                glosas_crudas = []
            
            # 4. Validar glosas contra el glosario
            glosas_validadas = []
            seen = set()
            for g in glosas_crudas:
                if g and isinstance(g, str):
                    g_clean = g.upper().strip()
                    if g_clean in glosario and g_clean not in seen:
                        glosas_validadas.append(g_clean)
                        seen.add(g_clean)
            
            glosas_de_palabras = glosas_validadas

        except Exception as e:
            logging.error(f"Error al invocar o procesar la cadena de LangChain: {e}")
            glosas_de_palabras = []
    
    # 5. Combinar glosas de palabras + números
    glosas = glosas_de_palabras + numeros_como_glosas
    
    conversational_history.append(texto_original)
    resultado = {"texto": texto_original, "glosas": glosas}
    logging.info(f"Resultado: {resultado}")
    
    # 6. Guardar en archivos (sincrónico)
    try:
        with open(TRANSCRIPCION_TXT, "a", encoding="utf-8") as f:
            f.write(str(resultado) + "\n")
        
        with open(TRANSCRIPCION_JSON, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(resultado)
            f.seek(0)
            f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
            with open(TRANSCRIPCION_JSON, "w", encoding="utf-8") as f:
                json.dump([resultado], f, ensure_ascii=False, indent=4)
    
    # 7. Enviar a FastAPI
    resultado_queue.put_nowait(resultado)


# =================================================================
# BLOQUE 4: FLUJO DE AUDIO (REVISADO - V7)
# =================================================================

# 1. Callback de audio (simple)
def audio_callback(indata, frames, time, status):
    """Pone bloques de audio en una cola."""
    if status:
        logging.warning(status)
    audio_queue.put(indata.copy())

# 2. Hilo de grabación
def record_audio():
    """Inicia la grabación de audio desde el micrófono."""
    global sistema_activo
    blocksize_frames = int(samplerate * block_duration_ms / 1000)
    try:
        with sd.InputStream(
            samplerate=samplerate, 
            channels=channels, 
            callback=audio_callback, 
            blocksize=blocksize_frames
        ):
            logging.info(f"🎙️ Grabando (bloques de {block_duration_ms}ms)...")
            while sistema_activo:
                sd.sleep(1000)
    except sd.PortAudioError as e:
        logging.critical(f"Error de PortAudio: {e}. ¿Micrófono en uso?")
        detener_sistema()
    except Exception as e:
        logging.error(f"Error en hilo de grabación: {e}")
        detener_sistema()

# 3. Hilo de Transcripción y Traducción (NUEVA LÓGICA V7)
def transcribe_and_process_pipeline():
    """
    Hilo único que maneja la transcripción y la traducción
    para evitar el cuello de botella de la cola.
    """
    global sistema_activo, audio_buffer
    
    vad_parameters = {
        "threshold": CONFIG['vad']['threshold'],
        "min_silence_duration_ms": CONFIG['vad']['min_silence_duration_ms']
    }
    context_frames = int(samplerate * context_duration_ms / 1000)
    previous_text = ""
    conversational_history = deque(maxlen=4)

    logging.info("Tubería de Transcripción-Traducción (V7) iniciada.")

    while sistema_activo:
        try:
            # 1. Acumular audio
            audio_chunk = audio_queue.get(timeout=0.1)
            audio_buffer = np.vstack((audio_buffer, audio_chunk))

            # 2. Si el búfer no es lo suficientemente grande, esperar más audio
            if len(audio_buffer) < frame_per_chunk:
                continue

            # 3. Tenemos 10 segundos, procesar
            audio_data_to_process = audio_buffer.flatten().astype(np.float32)
            audio_buffer = audio_buffer[-context_frames:] # Conservar contexto
            
            logging.info(f"Procesando chunk de {len(audio_data_to_process)/samplerate:.1f}s...")
            
            # 4. Transcribir el búfer
            segments, info = model.transcribe(
                audio_data_to_process,
                language="es",
                beam_size=CONFIG['whisper']['beam_size'],
                vad_filter=True,
                vad_parameters=vad_parameters,
                initial_prompt=previous_text
            )

            # 5. Bucle INTERNO: Procesar cada segmento INMEDIATAMENTE
            full_transcription_chunk = []
            for segment in segments:
                if not sistema_activo: # Chequeo de seguridad para salir rápido
                    break
                
                texto_transcrito = segment.text.strip()
                if (texto_transcrito and 
                    segment.no_speech_prob < CONFIG['whisper']['no_speech_prob']):
                    
                    logging.info(f"Frase detectada (VAD): '{texto_transcrito}'")
                    full_transcription_chunk.append(texto_transcrito)
                    
                    # --- ¡EL CAMBIO CLAVE! ---
                    # Llamar a la lógica de traducción aquí mismo,
                    # no en un hilo separado.
                    process_text_segment(texto_transcrito, conversational_history)
            
            # 6. Actualizar el prompt de Whisper para el siguiente chunk
            if full_transcription_chunk:
                previous_text = " ".join(full_transcription_chunk)
                logging.info(f"Contexto de prompt actualizado a: '...{previous_text[-50:]}'")
            else:
                logging.info("Audio procesado (VAD), sin habla detectada.")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error en la tubería de Transcripción-Traducción: {e}")
            time.sleep(1)


# =================================================================
# BLOQUE 5: CONTROL DEL SISTEMA
# =================================================================
def iniciar_sistema():
    global sistema_activo
    if sistema_activo:
        logging.warning("El sistema ya está iniciado.")
        return
        
    sistema_activo = True
    
    threads = [
        threading.Thread(target=record_audio, name="AudioRecorder", daemon=True),
        
        # --- ACTUALIZADO ---
        # Llama a la nueva tubería unificada
        threading.Thread(target=transcribe_and_process_pipeline, name="PipelineThread", daemon=True),
        
        # EL HILO 'process_text' YA NO ES NECESARIO
        # threading.Thread(target=process_text, name="TextProcessor", daemon=True)
    ]
    
    for t in threads:
        t.start()
        
    logging.info("🚀 Sistema de transcripción + glosas (Tubería V7) iniciado.")

def detener_sistema():
    global sistema_activo
    if not sistema_activo:
        logging.warning("El sistema ya estaba detenido.")
        return
        
    logging.info("🛑 Deteniendo sistema... por favor espere.")
    sistema_activo = False


if __name__ == "__main__":
    print("Este es un módulo de transcripción. No está diseñado para ejecutarse directamente.")
    print("Por favor, ejecute fastapi_interface.py en su lugar.")