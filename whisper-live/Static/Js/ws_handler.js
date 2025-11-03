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

    // ================== INICIO DE LA MEJORA ==================
    // 1. Creamos una cola para acumular mensajes y un temporizador.
    let messageQueue = [];
    let debounceTimer = null;
    const DEBOUNCE_DELAY_MS = 150; // Aumenta si aún ves saltos, disminuye para más reactividad.
    // =================== FIN DE LA MEJORA ====================


    // Conecta el WebSocket al cargar la página
    connect();

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;

        ws = new WebSocket("ws://" + window.location.host + "/ws");

        ws.onopen = () => {
            console.log("✅ WebSocket conectado y esperando órdenes.");
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            configBtn.disabled = false;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const items = Array.isArray(data) ? data : [data];
                
                // ================== INICIO DE LA MEJORA ==================
                // 2. En lugar de actualizar el HTML, añadimos los mensajes a la cola.
                messageQueue.push(...items);

                // 3. Reiniciamos el temporizador cada vez que llega un mensaje.
                clearTimeout(debounceTimer);

                // 4. Programamos la actualización del HTML para que ocurra una sola vez
                //    después de una breve pausa sin nuevos mensajes.
                debounceTimer = setTimeout(() => {
                    if (messageQueue.length > 0) {
                        actualizarResultadosEnHTML(messageQueue);
                        messageQueue = []; // Vaciamos la cola después de procesarla
                    }
                }, DEBOUNCE_DELAY_MS);
                // =================== FIN DE LA MEJORA ====================

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

    // ================== INICIO DE LA MEJORA ==================
    // 5. La lógica para actualizar el HTML ahora está en su propia función.
    function actualizarResultadosEnHTML(items) {
        if (pInicial && pInicial.style.display !== 'none') {
            pInicial.style.display = 'none';
        }

        // Usamos un DocumentFragment para una actualización más eficiente del DOM.
        const fragment = document.createDocumentFragment();

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
            fragment.appendChild(itemDiv);
        });

        // Añadimos todos los nuevos elementos al DOM de una sola vez.
        resultadosDiv.prepend(fragment);
    }
    // =================== FIN DE LA MEJORA ====================


    startBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'start'...");
            ws.send(JSON.stringify({ command: "start" }));
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