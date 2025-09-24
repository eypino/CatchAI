from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import signal

# Importar procesamiento desde Models
from Models import Whisper_transcribe as wp

app = FastAPI(title="CatchAI - Transcriptor en vivo")

# Carpeta de templates
templates = Jinja2Templates(directory="Templates")

sistema_iniciado = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sistema_iniciado
    await websocket.accept()

    if not sistema_iniciado:
        wp.iniciar_sistema()
        sistema_iniciado = True

    last_index = 0
    while True:
        if len(wp.resultados_globales) > last_index:
            for r in wp.resultados_globales[last_index:]:
                await websocket.send_json(r)
            last_index = len(wp.resultados_globales)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/stop")
def stop_app():
    wp.detener_sistema()
    return {"status": "Sistema detenido"}


@app.post("/exit")
def exit_app():
    os.kill(os.getpid(), signal.SIGTERM)  # Termina el proceso
    return {"status": "Apagando servidor..."}

if __name__ == "__main__":
    uvicorn.run("fastapi_interface:app", host="localhost", port=8000, reload=True)
