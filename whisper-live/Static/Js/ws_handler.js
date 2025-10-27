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
    const pInicial = resultadosDiv.querySelector("p"); // El párrafo inicial "La transcripción aparecerá aquí..."

    // Función principal para conectar el WebSocket
    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("El WebSocket ya está conectado.");
            return;
        }

        ws = new WebSocket("ws://" + window.location.host + "/ws");

        ws.onopen = () => {
            console.log("✅ WebSocket conectado!");
            // Limpiar el mensaje inicial si todavía está ahí
            if (pInicial) {
                pInicial.style.display = 'none';
            }
            // Actualizar estado de los botones
            startBtn.disabled = true;
            stopBtn.disabled = false;
            modelSelect.disabled = true;
            deviceSelect.disabled = true;
            computeSelect.disabled = true;
            configBtn.disabled = true;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const items = Array.isArray(data) ? data : [data];

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
                    
                    resultadosDiv.prepend(itemDiv); // prepend para que lo nuevo aparezca arriba
                });

            } catch (e) {
                console.error("Error procesando mensaje:", e);
            }
        };

        ws.onclose = () => {
            console.warn("🔌 WebSocket desconectado.");
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

    startBtn.addEventListener("click", connect);

    stopBtn.addEventListener("click", async () => {
        if (ws) {
            ws.close();
        }
        await fetch("/stop", { method: "POST" });
    });

    exitBtn.addEventListener("click", async () => {
        if (ws) {
            ws.close();
        }
        await fetch("/exit", { method: "POST" });
    });
});