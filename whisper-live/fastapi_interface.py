from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import signal
import asyncio
import queue

from Models.Whisper_transcribe_V2 import resultado_queue, iniciar_sistema, detener_sistema

app = FastAPI(title="CatchAI - Transcriptor en vivo")

templates = Jinja2Templates(directory="Templates")

sistema_iniciado = False
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sistema_iniciado
    await websocket.accept()

    if not sistema_iniciado:
        iniciar_sistema()
        sistema_iniciado = True


    try:
        while True:
            await asyncio.sleep(0.05)

            nuevos_resultados = []
            while not resultado_queue.empty():
                try:
                    resultado = resultado_queue.get_nowait()
                    nuevos_resultados.append(resultado)
                except queue.Empty:
                    break
            
            if nuevos_resultados:
                print(f"✔️ FastAPI: Enviando {len(nuevos_resultados)} resultados al frontend.")
                await websocket.send_json(nuevos_resultados)

    except WebSocketDisconnect:
        print("[WS] Cliente desconectado")
    except Exception as e:
        print(f"[WS] Error en WebSocket: {e}")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/stop")
def stop_app():
    detener_sistema()
    return {"status": "Sistema detenido"}


@app.post("/exit")
def exit_app():
    detener_sistema() 
    asyncio.sleep(0.5) 
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "Apagando servidor..."}


class ConfiguracionModelo(BaseModel):
    model_size: str
    device: str
    compute_type: str


@app.post("/configurar_modelo")
def configurar_modelo(cfg: ConfiguracionModelo):
    from Models import Whisper_transcribe_V2 as wp

    try:
        print(f"Intentando reconfigurar el modelo a: {cfg.model_size}, {cfg.device}, {cfg.compute_type}")
        wp.model = wp.WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        print("Reconfiguración del modelo exitosa.")
        return {"status": "ok", "modelo": cfg.model_size, "device": cfg.device, "compute_type": cfg.compute_type}
    except Exception as e:
        print(f"Error al reconfigurar el modelo: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("fastapi_interface:app", host="localhost", port=8000, reload=True)