// La regla de oro: Esperar a que todo el HTML esté cargado y listo.
document.addEventListener("DOMContentLoaded", () => {
    
    let ws;
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const exitBtn = document.getElementById("exitBtn");
    const resultadosDiv = document.getElementById("resultados");
    const modelSelect = document.getElementById("model-select");
    const deviceSelect = document.getElementById("device-select");
    const computeSelect = document.getElementById("compute-select");
    const configBtn = document.querySelector("button[onclick='configurarModelo()']");
    const pInicial = resultadosDiv.querySelector("p");

    // Conecta el WebSocket al cargar la página
    connect();

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;

        ws = new WebSocket("ws://" + window.location.host + "/ws");

        ws.onopen = () => {
            console.log("✅ WebSocket conectado y esperando órdenes.");
            // Al conectar, habilitamos el botón de start y deshabilitamos el de stop
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            configBtn.disabled = false;
        };

        ws.onmessage = (event) => {
            // ===== AÑADE ESTA LÍNEA PARA VERIFICAR =====
            console.log("Mensaje recibido del WebSocket:", event.data);
            try {
                const data = JSON.parse(event.data);
                const items = Array.isArray(data) ? data : [data];
                
                if (pInicial && pInicial.style.display !== 'none') {
                    pInicial.style.display = 'none';
                }

                items.forEach((segmento) => {
                    const itemDiv = document.createElement("div");
                    itemDiv.className = "transcripcion-item";
                    const textoP = document.createElement("p");
                    textoP.className = "texto-original";
                    textoP.textContent = `"${segmento.texto ?? "..."}"`;
                    const glosasDiv = document.createElement("div");
                    glosasDiv.className = "glosas-container";
                    if (segmento.glosas && segmento.glosas.length > 0) {
                        segmento.glosas.forEach(glosa => {
                            const glosaSpan = document.createElement("span");
                            glosaSpan.className = "glosa-pill";
                            glosaSpan.textContent = glosa;
                            glosasDiv.appendChild(glosaSpan);
                        });
                    }
                    itemDiv.appendChild(textoP);
                    itemDiv.appendChild(glosasDiv);
                    resultadosDiv.prepend(itemDiv);
                });
            } catch (e) {
                console.error("Error procesando mensaje:", e);
            }
        };

        ws.onclose = () => {
            console.warn("🔌 WebSocket desconectado.");
            // Se resetea el estado de los botones
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            configBtn.disabled = false;
        };

        ws.onerror = (error) => {
            console.error("❌ Error en WebSocket:", error);
            ws.close();
        };
    }

    startBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'start'...");
            ws.send(JSON.stringify({ command: "start" }));
            
            // Actualizar estado de los botones
            startBtn.disabled = true;
            stopBtn.disabled = false;
            modelSelect.disabled = true;
            deviceSelect.disabled = true;
            computeSelect.disabled = true;
            configBtn.disabled = true;
        }
    });

    stopBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'stop'...");
            ws.send(JSON.stringify({ command: "stop" }));

            // Actualizar estado de los botones
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            configBtn.disabled = false;
        }
    });

    exitBtn.addEventListener("click", async () => {
        if (ws) {
            ws.close();
        }
        await fetch("/exit", { method: "POST" });
    });
});