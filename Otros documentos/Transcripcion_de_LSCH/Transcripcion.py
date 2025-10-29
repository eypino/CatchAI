import re
import pandas as pd
from docx import Document

# Archivos DOCX (puedes añadir los dos tomos)
docs = ["Transcripcion_de_LSCH\Diccionario_LSCh_A-H-converted.docx", "Transcripcion_de_LSCH\Diccionario_LSCh_I-Z-converted.docx"]

texto = ""
for ruta_doc in docs:
    doc = Document(ruta_doc)
    for para in doc.paragraphs:
        contenido = para.text.strip()
        if contenido:
            texto += contenido + "\n"

# --- Regex para glosas ---
# Incluye:
# - Palabras MAYÚSCULAS (con acentos y Ñ)
# - Palabras compuestas con "-" o "/"
# - Glosas de varias palabras en MAYÚSCULAS
patron_glosas = r"\b(?:[A-ZÁÉÍÓÚÑ]+(?:[-/][A-ZÁÉÍÓÚÑ]+)*)+(?: [A-ZÁÉÍÓÚÑ]+(?:[-/][A-ZÁÉÍÓÚÑ]+)*)*\b"

glosas = re.findall(patron_glosas, texto)

# Eliminar duplicados y ordenar
glosas_unicas = sorted(set(glosas))

# Guardar CSV
df = pd.DataFrame(glosas_unicas, columns=["Palabra"])
df.to_csv("diccionario_lsch_glosas.csv", index=False, encoding="utf-8")

print(f"✅ Se extrajeron {len(glosas_unicas)} glosas únicas del diccionario LSCh.")