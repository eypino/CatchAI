import re
import pandas as pd

# Cargar el CSV existente
df = pd.read_csv("diccionario_lsch_final_revision.csv", dtype=str).fillna("")

nuevas_filas = []

for i, row in df.iterrows():
    sinonimos = row["Sinonimos"]
    palabra = row["Palabra"]

    # Detectar si hay una segunda entrada incrustada
    # Ejemplo: "Cartón, cartulina. Construcción... Esp.: sust. f. Casa, vivienda..."
    match = re.search(r"\.\s*Esp\.\s*:\s*(sust\.|v\.)", sinonimos)
    if match:
        # Cortar el texto antes y después del nuevo bloque
        parte_1 = sinonimos[:match.start()].strip()
        parte_2 = sinonimos[match.start():].strip()

        # Buscar la palabra nueva en la parte posterior (después de 'Esp.: ...')
        palabra_nueva = None
        m2 = re.search(r"Esp\.\s*:\s*(?:sust\.|v\.)\s*(?:[mf]\.\s*)?([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ]+)", parte_2)
        if m2:
            palabra_nueva = m2.group(1).upper()

        # Actualizar el registro original (dejando solo su parte válida)
        df.at[i, "Sinonimos"] = parte_1

        if palabra_nueva:
            nuevas_filas.append({
                "Palabra": palabra_nueva,
                "Descripcion": parte_2,
                "Categoria": "",
                "Sinonimos": "",
                "Antonimos": ""
            })

# Agregar nuevas filas al final
if nuevas_filas:
    print(f"🔍 Se detectaron {len(nuevas_filas)} posibles entradas embebidas.")
    df = pd.concat([df, pd.DataFrame(nuevas_filas)], ignore_index=True)
else:
    print("✅ No se detectaron entradas embebidas.")

# Guardar nueva versión
df.to_csv("diccionario_lsch_reparado.csv", index=False, encoding="utf-8-sig")
print("✅ Guardado: diccionario_lsch_reparado.csv")
