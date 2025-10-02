# fastapi_interface.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import signal
import asyncio
import json

# Importa tu pipeline Whisper
from Models import Whisper_transcribe as wp

app = FastAPI(title="CatchAI - Transcriptor en vivo")

# Templates (tu index.html está en ./Templates/index.html)
templates = Jinja2Templates(directory="Templates")

sistema_iniciado = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para Godot:
    - Inicializa el sistema (solo una vez).
    - Lee 'wp.resultados_globales' y envía SOLO lo nuevo.
    - Siempre envía en formato JSON de ARRAY, p. ej.:
        [
          {"inicio":0.0,"fin":2.0,"texto":"...","glosas":["H","O","L","A"]},
          ...
        ]
    - Si 'glosas' viene como string (p. ej. "['H','O','L','A']"),
      se convierte a lista antes de enviar.
    """
    global sistema_iniciado
    await websocket.accept()

    if not sistema_iniciado:
        wp.iniciar_sistema()
        sistema_iniciado = True

    last_index = 0

    def _coerce_segments(raw_list):
        fixed = []
        for r in raw_list:
            # Si todo el segmento viene como string JSON
            if isinstance(r, str):
                try:
                    r = json.loads(r)
                except Exception:
                    continue

            # Asegurar diccionario
            if not isinstance(r, dict):
                continue

            # Asegurar que 'glosas' sea lista
            g = r.get("glosas", [])
            if isinstance(g, str):
                # Intenta parsear "['A','B']" → ["A","B"]
                try:
                    g2 = json.loads(g.replace("'", '"'))
                    if isinstance(g2, list):
                        r["glosas"] = g2
                    else:
                        r["glosas"] = []
                except Exception:
                    r["glosas"] = []
            elif not isinstance(g, list):
                r["glosas"] = []

            fixed.append(r)
        return fixed

    try:
        while True:
            await asyncio.sleep(0.05)  # evita busy loop
            current_len = len(wp.resultados_globales)
            if current_len > last_index:
                nuevos_raw = wp.resultados_globales[last_index:current_len]
                nuevos = _coerce_segments(nuevos_raw)
                if nuevos:
                    await websocket.send_json(nuevos)  # Godot recibe Array de segmentos
                last_index = current_len
    except WebSocketDisconnect:
        print("[WS] Cliente desconectado")
    except Exception as e:
        print(f"[WS] Error: {e}")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/stop")
def stop_app():
    wp.detener_sistema()
    return {"status": "Sistema detenido"}


@app.post("/exit")
def exit_app():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "Apagando servidor..."}


# --- Opcional: endpoint de prueba para empujar datos al WS sin Whisper ---
@app.get("/demo")
async def demo_ws():
    demo = [
        {
            "inicio": 0.0,
            "fin": 2.0,
            "texto": "hola probando",
            "glosas": ["HOLA", "P", "R", "O", "B", "A", "N", "D", "O"],
        }
    ]
    # simula que Whisper agregó resultados
    if not hasattr(wp, "resultados_globales") or wp.resultados_globales is None:
        wp.resultados_globales = []
    wp.resultados_globales.extend(demo)
    return {"ok": True, "count": len(demo)}


if __name__ == "__main__":
    # Requisitos: pip install "uvicorn[standard]" websockets
    uvicorn.run("fastapi_interface:app", host="localhost", port=8000, reload=True)
