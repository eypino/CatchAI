import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_TOKEN = os.path.join(SCRIPT_DIR, "github_token.txt")

def obtener_token():
    # Si ya existe un archivo guardado, lo lee
    if os.path.exists(RUTA_TOKEN):
        with open(RUTA_TOKEN, "r") as f:
            token = f.read().strip()
            if token:
                return token

    # Si no existe, solicita el token al usuario
    print("⚙️  No se encontró el token de GitHub Models.")
    token = input("👉 Ingrese su token personal de GitHub (formato ghp_...): ").strip()

    # Guarda el token localmente
    with open(RUTA_TOKEN, "w") as f:
        f.write(token)
    print("✅ Token guardado correctamente en github_token.txt (no se sube a GitHub).")
    return token