# ===================== EXISTENTE (tuyo) =====================
from pydantic import BaseModel 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import signal
import asyncio

from Models.Whisper_transcribe_V4 import resultado_queue, iniciar_sistema, detener_sistema, sistema_activo

# === NUEVO (Marcos): imports solo para el modo de prueba desde archivo ===
import json  # NUEVO
from pathlib import Path  # NUEVO
# ==========================================================================

app = FastAPI(title="CatchAI - Transcriptor en vivo")
templates = Jinja2Templates(directory="Templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
sender_task = None

# ===================== (MARCOS) NUEVO: soporte broadcast =====================
# En lugar de un solo "consumidor" por WebSocket, mantenemos una lista de
# clientes conectados y un loop que saca de resultado_queue y emite a todos.
active_clients: set[WebSocket] = set()
broadcaster_task: asyncio.Task | None = None

async def _safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_json(payload)
    except Exception:
        # si el cliente se cayó silenciosamente, lo limpiaremos en el endpoint
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


# ===================== EXISTENTE =====================
async def send_results(websocket: WebSocket):
    """Tarea que se encarga de enviar resultados de la cola al frontend."""
    try:
        while True:
            resultado = await resultado_queue.get()

            # ===================== CAMBIO (MARCOS) =====================
            # Antes enviabas una LISTA con un elemento: await websocket.send_json([resultado])
            # Godot ahora espera un OBJETO JSON (diccionario), no lista.
            await websocket.send_json(resultado)
            # ===========================================================

            resultado_queue.task_done()

    except (WebSocketDisconnect, asyncio.CancelledError):
        print(" Tarea de envío de resultados detenida.")
    except Exception as e:
        print(f"❌ Error en la tarea de envío: {e}")


# ===================== EXISTENTE =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sender_task
    # ===== (MARCOS) registrar cliente para broadcast =====
    global broadcaster_task
    await websocket.accept()
    active_clients.add(websocket)  # (MARCOS)
    print("✔️ Cliente conectado. Esperando órdenes...")

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "start":
                if not sistema_activo:
                    print("🚀 Recibido comando 'start'. Iniciando sistema...")
                    iniciar_sistema()
                    # ===== (MARCOS) iniciar broadcaster si no existe =====
                    if broadcaster_task is None or broadcaster_task.done():
                        broadcaster_task = asyncio.create_task(broadcast_loop())
                else:
                    print("⚠️ Sistema ya estaba iniciado. Comando 'start' ignorado.")
                # ===== (MARCOS) ya no lanzamos send_results por-cliente =====
                # sender_task = asyncio.create_task(send_results(websocket))

            elif command == "stop":
                if sistema_activo:
                    print("🛑 Recibido comando 'stop'. Deteniendo sistema...")
                    detener_sistema()
                    # (MARCOS) mantenemos el broadcaster vivo; si la cola queda vacía, esperará
                    if sender_task:
                        sender_task.cancel()
                        sender_task = None
                else:
                    print("⚠️ Sistema ya estaba detenido. Comando 'stop' ignorado.")

            # === NUEVO (modo prueba desde archivo) =====================
            elif command == "start_file":
                print("▶️ Modo prueba: enviando Storage/transcripciones.json")
                await stream_transcripciones_desde_archivo(websocket)
            # ===========================================================

            # === PING (conectividad) ==================================
            elif command == "ping":
                # ===================== CAMBIO (ChatGPT) =================
                # Antes: await websocket.send_json([{"ok": True, "msg": "pong"}])
                await websocket.send_json({"ok": True, "msg": "pong"})
                # ========================================================

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado.")
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        # ===== (MARCOS) remover cliente del set de broadcast =====
        if websocket in active_clients:
            active_clients.remove(websocket)
        if sistema_activo and not active_clients:
            # Opcional: si no quedan clientes, puedes detener el sistema
            # detener_sistema()
            pass
        if sender_task:
            sender_task.cancel()
            sender_task = None
        await websocket.close()


# ===================== EXISTENTE  =====================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===================== CAMBIO (Marcos) =====================
# Tu ruta era sincrónica pero llamaba a asyncio.sleep(), que no tiene efecto.
# Puedes elegir una de estas dos versiones. Dejamos la ASÍNCRONA activa.

@app.post("/exit")
async def exit_app():
    detener_sistema()
    await asyncio.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "Apagando servidor..."}

# Si prefieres mantenerla síncrona, usa esta en su lugar:
# @app.post("/exit")
# def exit_app_sync():
#     import time
#     detener_sistema()
#     time.sleep(0.5)
#     os.kill(os.getpid(), signal.SIGTERM)
#     return {"status": "Apagando servidor..."}
# ===========================================================


# ===================== EXISTENTE =====================
class ConfiguracionModelo(BaseModel):
    model_size: str
    device: str
    compute_type: str

@app.post("/configurar_modelo")
def configurar_modelo(cfg: ConfiguracionModelo):
    from Models import Whisper_transcribe_V4 as wp
    try:
        print(f"Intentando reconfigurar el modelo a: {cfg.model_size}, {cfg.device}, {cfg.compute_type}")
        wp.model = wp.WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        print("Reconfiguración del modelo exitosa.")
        return {"status": "ok", "modelo": cfg.model_size, "device": cfg.device, "compute_type": cfg.compute_type}
    except Exception as e:
        print(f"Error al reconfigurar el modelo: {e}")
        return {"status": "error", "message": str(e)}


# ===================== EXISTENTE =====================
if __name__ == "__main__":
    uvicorn.run("fastapi_interface:app", host="localhost", port=8000, reload=True)


# ============================ BLOQUE AÑADIDO (Marcos) ============================
# Función de apoyo para pruebas: envía por WebSocket el contenido de
# Storage/transcripciones.json (un arreglo de objetos con campos 'texto' y 'glosas').
# Ahora enviamos CADA OBJETO directamente (no dentro de lista) para que Godot lo parsee.
async def stream_transcripciones_desde_archivo(websocket: WebSocket):
    try:
        base_dir = Path(__file__).parent
        trans_path = base_dir / "Storage" / "transcripciones.json"
        if not trans_path.exists():
            await websocket.send_json({"error": "transcripciones.json no encontrado"})
            return

        with trans_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Enviar como objetos individuales (no lista)
        for item in data if isinstance(data, list) else [data]:
            await websocket.send_json(item)
            await asyncio.sleep(0.05)  # espaciamiento opcional
        print("✅ Fin de envío de transcripciones.json")
    except Exception as e:
        print(f"❌ Error enviando transcripciones desde archivo: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
# ================================================================================
