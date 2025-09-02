import pandas as pd

# Cargar el archivo original
df = pd.read_csv("diccionario_lsch_glosas.csv")

# Lista de palabras a evaluar automáticamente (artículos, preposiciones, etc.)
stopwords = {"AL", "AS", "DE", "DEL", "EN", "EL", "LA", "LO", "LAS", "LOS",
             "SE", "SU", "UN", "UNA", "Y", "O", "II"}

# Candidatas: palabras cortas o que están en stopwords
candidatas = df[df["Palabra"].apply(lambda x: len(x) <= 2 or x in stopwords)]

# Lista final de palabras aprobadas
palabras_limpias = []

for palabra in df["Palabra"]:
    if palabra in candidatas["Palabra"].values:
        # Preguntar al usuario
        decision = input(f"⚠️ La palabra '{palabra}' puede ser ruido. ¿Quieres conservarla? (s/n): ")
        if decision.lower() == "s":
            palabras_limpias.append(palabra)
        else:
            print(f"❌ '{palabra}' eliminada.")
    else:
        palabras_limpias.append(palabra)

# Crear nuevo DataFrame limpio
df_limpio = pd.DataFrame(sorted(set(palabras_limpias)), columns=["Palabra"])
df_limpio.to_csv("diccionario_lsch_glosas_limpio.csv", index=False, encoding="utf-8")

print(f"✅ Proceso completado. Se guardó 'diccionario_lsch_glosas_limpio.csv' con {len(df_limpio)} palabras.")
