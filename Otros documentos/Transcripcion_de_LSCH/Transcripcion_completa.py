import re
import pandas as pd
from docx import Document

# --- FUNCIÓN DE PROCESAMIENTO FINAL Y ROBUSTA ---
def procesar_entrada(palabra, lineas_cuerpo):
    """
    Procesa una entrada extrayendo secuencialmente cada parte del texto.
    Este método es el más fiable para formatos inconsistentes.
    """
    if not palabra or not lineas_cuerpo:
        return None

    # PASO 1: Unir todo en una sola cadena de texto y limpiar espacios
    cuerpo = " ".join(lineas_cuerpo).strip()
    cuerpo = re.sub(r"\s+", " ", cuerpo)

    # Inicializar todos los campos
    descripcion = ""
    categoria = ""
    sinonimos = ""
    antonimos = ""

    # PASO 2: Extraer y eliminar los antónimos
    ant_match = re.search(r"\bAnt\.\s*[:\s]*(.*)", cuerpo, re.IGNORECASE)
    if ant_match:
        antonimos = ant_match.group(1).strip()
        cuerpo = cuerpo[:ant_match.start()].strip()

    # PASO 3: Extraer y eliminar el bloque "Esp.:"
    esp_texto = ""
    esp_match = re.search(r"Esp\.\s*:\s*(.*)", cuerpo, re.IGNORECASE)
    if esp_match:
        esp_texto = esp_match.group(1).strip()
        cuerpo = cuerpo[:esp_match.start()].strip()

    # PASO 4: Lo que queda es la descripción
    descripcion = cuerpo

    # PASO 5: Analizar el bloque Esp.
    if esp_texto:
        cat_regex = r"^((?:(?:v|sust|adj|adv|m|f|tr|intr|prnl)\.\s*|o\s*|/\s*)+)"
        cat_match = re.search(cat_regex, esp_texto, re.IGNORECASE)

        if cat_match:
            categoria = cat_match.group(1).strip()
            sinonimos = esp_texto[cat_match.end():].strip()
            sinonimos = re.sub(r"^[.,\s]+", "", sinonimos)
        else:
            sinonimos = esp_texto

    return {
        "Palabra": palabra,
        "Descripcion": descripcion,
        "Categoria": categoria,
        "Sinonimos": sinonimos,
        "Antonimos": antonimos
    }


# --- SCRIPT PRINCIPAL ---
docs = [
    "Diccionario_LSCh_A-H-converted.docx",
    "Diccionario_LSCh_I-Z-converted.docx"
]

filas = []
entrada_actual = {"palabra": None, "cuerpo": []}
regex_palabra = r"^[A-ZÁÉÍÓÚÑ0-9\-/]+(?:/[A-ZÁÉÍÓÚÑ0-9\-/]+)*$"

for ruta_doc in docs:
    doc = Document(ruta_doc)
    for para in doc.paragraphs:
        linea = para.text.strip()
        if not linea:
            continue

        # Detecta inicio de palabra nueva
        if re.fullmatch(regex_palabra, linea):
            if entrada_actual["palabra"]:
                resultado = procesar_entrada(entrada_actual["palabra"], entrada_actual["cuerpo"])
                if resultado:
                    filas.append(resultado)
            entrada_actual = {"palabra": linea, "cuerpo": []}
        else:
            if entrada_actual["palabra"]:
                entrada_actual["cuerpo"].append(linea)

# Procesar la última entrada
if entrada_actual["palabra"]:
    resultado = procesar_entrada(entrada_actual["palabra"], entrada_actual["cuerpo"])
    if resultado:
        filas.append(resultado)

# --- CREAR DATAFRAME FINAL ---
df = pd.DataFrame(filas)
columnas_ordenadas = ["Palabra", "Descripcion", "Categoria", "Sinonimos", "Antonimos"]
for col in columnas_ordenadas:
    if col not in df.columns:
        df[col] = ""
df = df[columnas_ordenadas]
df = df.drop_duplicates(subset=["Palabra"])

# --- GUARDAR CSV PRINCIPAL ---
df.to_csv("diccionario_lsch_final_OK.csv", index=False, encoding="utf-8-sig")
print(f"✅ Exportadas {len(df)} entradas del diccionario LSCh.")

# --- DETECCIÓN DE ENTRADAS INCOMPLETAS ---
cond_incompletas = (
    (df["Descripcion"].str.strip() == "") |
    (df["Sinonimos"].str.strip() == "")
)
df_incompletas = df[cond_incompletas]

if not df_incompletas.empty:
    df_incompletas.to_csv("diccionario_incompletas.csv", index=False, encoding="utf-8-sig")
    print(f"⚠️ Se detectaron {len(df_incompletas)} entradas con problemas (sin descripción o sin sinónimos).")
    print("🔎 Ejemplo de las primeras 10 problemáticas:")
    print(df_incompletas.head(10))
else:
    print("✅ Todas las entradas tienen descripción y al menos un sinónimo.")
