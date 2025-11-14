from pydantic import BaseModel 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import signal
import asyncio

# ================== ¡CAMBIO CLAVE! ==================
# Ya no importamos 'sistema_activo'.
from Models.Whisper_transcribe_V4 import resultado_queue, iniciar_sistema, detener_sistema
# ================== FIN CAMBIO ==================

import json
from pathlib import Path

app = FastAPI(title="CatchAI - Transcriptor en vivo")
templates = Jinja2Templates(directory="Templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
sender_task = None 

# ===================== (MARCOS) NUEVO: soporte broadcast =====================
active_clients: set[WebSocket] = set()
broadcaster_task: asyncio.Task | None = None

async def _safe_send(ws: WebSocket, payload: dict):
    try:
        # Tu JS (ws_handler.js) espera una LISTA, así que la envolvemos
        await ws.send_json([payload])
    except Exception:
        pass

async def broadcast_loop():
    """Consume resultado_queue UNA VEZ y envía cada item a TODOS los clientes."""
    while True:
        data = await resultado_queue.get()
        if active_clients:
            await asyncio.gather(
                *[_safe_send(ws, data) for ws in list(active_clients)],
                return_exceptions=True
            )
        resultado_queue.task_done()
# ============================================================================


# (Esta función 'send_results' parece ya no ser usada si 'broadcast_loop' está activo,
#  pero la dejamos por si acaso)
async def send_results(websocket: WebSocket):
    """Tarea que se encarga de enviar resultados de la cola al frontend."""
    try:
        while True:
            resultado = await resultado_queue.get()
            await websocket.send_json([resultado]) # Enviamos como lista
            resultado_queue.task_done()
    except (WebSocketDisconnect, asyncio.CancelledError):
        print(" Tarea de envío de resultados detenida.")
    except Exception as e:
        print(f"❌ Error en la tarea de envío: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sender_task
    global broadcaster_task
    await websocket.accept()
    active_clients.add(websocket)
    print("✔️ Cliente conectado. Esperando órdenes...")

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            # ================== ¡LÓGICA SIMPLIFICADA V14! ==================
            if command == "start":
                print("🚀 Recibido comando 'start'.")
                iniciar_sistema() # El módulo V4 maneja su propio estado
                
                # Iniciar el broadcaster de Marcos
                if broadcaster_task is None or broadcaster_task.done():
                    broadcaster_task = asyncio.create_task(broadcast_loop())

            elif command == "stop":
                print("🛑 Recibido comando 'stop'.")
                detener_sistema() # El módulo V4 maneja su propio estado
            # ================== FIN LÓGICA V14 ==================

            elif command == "start_file":
                print("▶️ Modo prueba: enviando Storage/transcripciones.json")
                await stream_transcripciones_desde_archivo(websocket)

            elif command == "ping":
                # Enviamos un objeto, no una lista
                await websocket.send_json({"ok": True, "msg": "pong"}) 

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado.")
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        if websocket in active_clients:
            active_clients.remove(websocket)
        
        # Si no quedan clientes, detenemos el sistema
        if not active_clients: # (Eliminada la comprobación de 'sistema_activo')
            print("🛑 Último cliente desconectado, deteniendo sistema.")
            detener_sistema()
            
            if broadcaster_task:
                 broadcaster_task.cancel()
                 broadcaster_task = None
             
        if sender_task: # Limpieza de la tarea antigua
            sender_task.cancel()
            sender_task = None
        
        # ¡No llames a websocket.close() aquí!
        

# ================== ¡RUTAS CORREGIDAS (V14)! ==================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # La ruta raíz "/" ahora sirve la APP de transcripción
    # (¡Asegúrate de que 'index.html' existe en 'Templates'!)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/main", response_class=HTMLResponse)
def main_page(request: Request):
    # La nueva ruta "/main" sirve la landing page
    return templates.TemplateResponse("main_page.html", {"request": request})
# ================== FIN RUTAS ==================


@app.post("/exit")
async def exit_app():
    detener_sistema()
    await asyncio.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "Apagando servidor..."}


class ConfiguracionModelo(BaseModel):
    model_size: str
    device: str
    compute_type: str

@app.post("/configurar_modelo")
def configurar_modelo(cfg: ConfiguracionModelo):
    from Models.Whisper_transcribe_V4 import model as wp_model, WhisperModel
    try:
        print(f"Intentando reconfigurar el modelo a: {cfg.model_size}, {cfg.device}, {cfg.compute_type}")
        wp_model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        print("Reconfiguración del modelo exitosa. (¡Recuerda reiniciar el stream!)")
        return {"status": "ok", "modelo": cfg.model_size, "device": cfg.device, "compute_type": cfg.compute_type}
    except Exception as e:
        print(f"Error al reconfigurar el modelo: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("fastapi_interface:app", host="localhost", port=8000, reload=True)


async def stream_transcripciones_desde_archivo(websocket: WebSocket):
    try:
        base_dir = Path(__file__).parent
        trans_path = base_dir / "Storage" / "transcripciones.json"
        if not trans_path.exists():
            await websocket.send_json({"error": "transcripciones.json no encontrado"})
            return

        with trans_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Enviar como objetos individuales
        for item in data if isinstance(data, list) else [data]:
            await websocket.send_json([item]) # Tu JS espera una lista
            await asyncio.sleep(0.05)
        print("✅ Fin de envío de transcripciones.json")
    except Exception as e:
        print(f"❌ Error enviando transcripciones desde archivo: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass