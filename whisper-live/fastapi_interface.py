from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import signal
import asyncio

from Models.Whisper_transcribe_V3 import resultado_queue, iniciar_sistema, detener_sistema, sistema_activo

app = FastAPI(title="CatchAI - Transcriptor en vivo")
templates = Jinja2Templates(directory="Templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
sender_task = None

async def send_results(websocket: WebSocket):
    """Tarea que se encarga de enviar resultados de la cola al frontend."""
    try:
        while True:
            resultado = await resultado_queue.get()
            
            await websocket.send_json([resultado])
            
            resultado_queue.task_done()

    except (WebSocketDisconnect, asyncio.CancelledError):
        print(" Tarea de envío de resultados detenida.")
    except Exception as e:
        print(f"❌ Error en la tarea de envío: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sender_task
    await websocket.accept()
    print("✔️ Cliente conectado. Esperando órdenes...")

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "start":
                if not sistema_activo:
                    print("🚀 Recibido comando 'start'. Iniciando sistema...")
                    iniciar_sistema()
                    sender_task = asyncio.create_task(send_results(websocket))
                else:
                    print("⚠️ Sistema ya estaba iniciado. Comando 'start' ignorado.")

            elif command == "stop":
                if sistema_activo:
                    print("🛑 Recibido comando 'stop'. Deteniendo sistema...")
                    detener_sistema()
                    if sender_task:
                        sender_task.cancel()
                        sender_task = None
                else:
                    print("⚠️ Sistema ya estaba detenido. Comando 'stop' ignorado.")

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado.")
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        if sistema_activo:
            print("🛑 Deteniendo sistema por desconexión.")
            detener_sistema()
        if sender_task:
            sender_task.cancel()
            sender_task = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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