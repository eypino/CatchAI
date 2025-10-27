import re
import pandas as pd

# === Ruta del CSV ya existente ===
archivo_entrada = "diccionario_lsch_reparado.csv"
archivo_salida = "diccionario_lsch_limpio.csv"
archivo_complemento = "diccionario_lsch_extraidas.csv"

# === Cargar dataset original ===
df = pd.read_csv(archivo_entrada, encoding="utf-8", sep=",", quotechar='"', on_bad_lines="skip", engine="python")
# === Patrón para detectar nuevas palabras dentro de las columnas ===
patron_esp = re.compile(
    r"Esp\.\s*:\s*((?:v|sust|adj|adv|m|f|tr|intr|prnl|pron)\.[^A-ZÁÉÍÓÚÑ]*)?\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑüÜñÑ0-9 ,.\-/]*)",
    re.IGNORECASE
)

# === Almacenar nuevas filas extraídas ===
nuevas_filas = []

# === Procesar cada fila del dataset ===
for i, fila in df.iterrows():
    descripcion = str(fila["Descripcion"])
    sinonimos = str(fila["Sinonimos"])
    antonimos = str(fila["Antonimos"])

    texto_total = " ".join([descripcion, sinonimos, antonimos])
    texto_total = re.sub(r"\s+", " ", texto_total).strip()

    # Buscar bloques "Esp.:"
    for match in re.finditer(patron_esp, texto_total):
        categoria = (match.group(1) or "").strip()
        sinonimos_nuevos = (match.group(2) or "").strip()

        # Intentar identificar la palabra principal dentro de esa parte
        palabra_match = re.search(r"\b([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\-/]*)", sinonimos_nuevos)
        palabra_nueva = palabra_match.group(1).upper() if palabra_match else None

        if palabra_nueva:
            nuevas_filas.append({
                "Palabra": palabra_nueva,
                "Descripcion": "",
                "Categoria": categoria,
                "Sinonimos": sinonimos_nuevos,
                "Antonimos": ""
            })

            # Remover esa parte del texto original para limpiar la fila base
            texto_total = texto_total[:match.start()].strip()

    # Actualizar la fila original limpia (solo descripción válida)
    partes = texto_total.split(" Esp.:")
    df.at[i, "Descripcion"] = partes[0].strip()
    df.at[i, "Sinonimos"] = "" if len(partes) > 1 else fila["Sinonimos"]
    df.at[i, "Antonimos"] = "" if len(partes) > 1 else fila["Antonimos"]

# === Crear DataFrame con las nuevas filas extraídas ===
df_extra = pd.DataFrame(nuevas_filas)

# === Eliminar duplicados y exportar ===
df_extra = df_extra.drop_duplicates(subset=["Palabra"])
df = df.drop_duplicates(subset=["Palabra"])

df.to_csv(archivo_salida, index=False, encoding="utf-8-sig")
df_extra.to_csv(archivo_complemento, index=False, encoding="utf-8-sig")

print(f"✅ Limpieza completada.")
print(f" - Archivo limpio: {archivo_salida}")
print(f" - Nuevas entradas detectadas: {len(df_extra)} (guardadas en {archivo_complemento})")
