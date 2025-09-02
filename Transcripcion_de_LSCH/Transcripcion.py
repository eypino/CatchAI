import pdfplumber
import re
import pandas as pd

pdfs = ["Diccionario_LSCh_A-H.pdf", "Diccionario_LSCh_I-Z.pdf"]
texto = ""

for ruta_pdf in pdfs:
    with pdfplumber.open(ruta_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages):
            try:
                # Aumentamos tolerancia para evitar cortes raros de texto
                contenido = pagina.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                if contenido:
                    texto += contenido + "\n"
            except Exception as e:
                print(f"⚠️ Error en página {i+1} de {ruta_pdf}: {e}")

# Extraer glosas: mayúsculas de 2+ letras, incluyendo acentos/Ñ
glosas = re.findall(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b", texto)
glosas_unicas = sorted(set(glosas))

# Guardar CSV
df = pd.DataFrame(glosas_unicas, columns=["Palabra"])
df.to_csv("diccionario_lsch_glosas.csv", index=False, encoding="utf-8")

print(f"✅ Se extrajeron {len(glosas_unicas)} palabras únicas del diccionario LSCh.")

