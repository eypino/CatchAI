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

# === NUEVO (ChatGPT): imports solo para el modo de prueba desde archivo ===
import json  # NUEVO
from pathlib import Path  # NUEVO
# ==========================================================================

app = FastAPI(title="CatchAI - Transcriptor en vivo")
templates = Jinja2Templates(directory="Templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
sender_task = None

async def send_results(websocket: WebSocket):
    """Tarea que se encarga de enviar resultados de la cola al frontend."""
    try:
        while True:
            resultado = await resultado_queue.get()

            # El cliente de Godot espera un ARRAY de objetos; mantenemos formato:
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

            # === NUEVO (ChatGPT): modo de prueba: enviar transcripciones.json por WS ===
            elif command == "start_file":
                # Envía el contenido de Storage/transcripciones.json como si fuera el stream
                print("▶️ Modo prueba: enviando Storage/transcripciones.json")
                await stream_transcripciones_desde_archivo(websocket)
            # ==========================================================================

            # === NUEVO (ChatGPT): util simple para probar conectividad desde Godot ===
            elif command == "ping":
                await websocket.send_json([{"ok": True, "msg": "pong"}])
            # ==========================================================================

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



# ============================ BLOQUE AÑADIDO (ChatGPT) ============================
# Función de apoyo para pruebas: envía por WebSocket el contenido de
# Storage/transcripciones.json (un arreglo de objetos con campos 'texto' y 'glosas').
# No interfiere con el flujo normal basado en la cola 'resultado_queue'.
async def stream_transcripciones_desde_archivo(websocket: WebSocket):
    try:
        base_dir = Path(__file__).parent
        trans_path = base_dir / "Storage" / "transcripciones.json"
        if not trans_path.exists():
            await websocket.send_json([{"error": "transcripciones.json no encontrado"}])
            return

        with trans_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Garantizamos enviar SIEMPRE como lista de un elemento (compatibilidad con Godot)
        for item in data if isinstance(data, list) else [data]:
            await websocket.send_json([item])
            await asyncio.sleep(0.05)  # pequeño espaciamiento opcional
        print("✅ Fin de envío de transcripciones.json")
    except Exception as e:
        print(f"❌ Error enviando transcripciones desde archivo: {e}")
        try:
            await websocket.send_json([{"error": str(e)}])
        except Exception:
            pass
# ================================================================================ 
