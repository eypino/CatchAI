from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
import json
import os

# ==== Configuración del cliente con GitHub ====
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=os.environ.get("GITHUB_TOKEN")  # 🔑 Usa tu token GitHub
)

# ==== Rutas ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STORAGE_DIR = os.path.join(BASE_DIR, "Storage")

# ==== Cargar glosario ====
GLOSARIO_PATH = os.path.join(STORAGE_DIR, "diccionario_lsch_glosas.csv")
glosario_df = pd.read_csv(GLOSARIO_PATH)
glosario = glosario_df["Palabra"].str.upper().str.strip().tolist()

# ==== Archivos persistentes ====
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "glosario.index")
NPY_LABELS_PATH = os.path.join(STORAGE_DIR, "glosario_labels.npy")
NPY_EMB_PATH = os.path.join(STORAGE_DIR, "glosario_embeddings.npy")  # 👈 nuevo archivo

# ==== Crear o cargar índice FAISS ====
if all(os.path.exists(p) for p in [FAISS_INDEX_PATH, NPY_LABELS_PATH, NPY_EMB_PATH]):
    print("📂 Cargando FAISS, labels y embeddings desde archivo...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    glosario = np.load(NPY_LABELS_PATH, allow_pickle=True).tolist()
    embeddings = np.load(NPY_EMB_PATH)
else:
    print("⚡ Generando embeddings del glosario (una sola vez)...")
    embeddings = []
    batch_size = 100
    for i in range(0, len(glosario), batch_size):
        batch = glosario[i:i+batch_size]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    glosario_embeddings = np.array(embeddings).astype("float32")

    # Crear índice FAISS
    dim = glosario_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(glosario_embeddings)

    # Guardar índice, embeddings y labels
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(NPY_LABELS_PATH, np.array(glosario))
    np.save(NPY_EMB_PATH, glosario_embeddings)
    print(f"✅ Índice FAISS guardado en {FAISS_INDEX_PATH}")
    print(f"✅ Labels guardados en {NPY_LABELS_PATH}")
    print(f"✅ Embeddings guardados en {NPY_EMB_PATH}")

# ==== Función para buscar glosas ====
def buscar_glosas(oracion, top_k=20):
    """Busca las glosas más cercanas semánticamente a una oración usando FAISS"""
    emb = client.embeddings.create(model="text-embedding-3-small", input=oracion).data[0].embedding
    emb = np.array([emb]).astype("float32")
    D, I = index.search(emb, top_k)
    return [glosario[i] for i in I[0]]

# ==== Cargar transcripciones ====
TRANSCRIPCION_JSON = os.path.join(STORAGE_DIR, "transcripciones.json")
with open(TRANSCRIPCION_JSON, "r", encoding="utf-8") as f:
    transcripciones = json.load(f)

resultados = []

# ==== Procesar ====
for entrada in transcripciones:
    texto = entrada.get("texto", "")

    # 1. Buscar glosas candidatas
    candidatas = buscar_glosas(texto, top_k=30)

    # 2. Pedir a GPT que seleccione glosas
    prompt = f"""
Convierte la siguiente oración a glosas LSCh.
Usa SOLO glosas de la lista candidata.

Oración:
{texto}

Glosas candidatas:
{", ".join(candidatas)}

Reglas:
- Nunca devuelvas deletreo.
- Devuelve SOLO un JSON con las claves: texto (original) y glosas (lista).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un traductor de texto a glosas de lengua de señas chilena."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    salida = json.loads(response.choices[0].message.content)
    resultados.append(salida)

# ==== Guardar resultado final ====
OUTPUT_JSON = os.path.join(STORAGE_DIR, "transcripciones_glosas.json")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=4)

print(f"✅ Traducción completada y guardada en {OUTPUT_JSON}")
