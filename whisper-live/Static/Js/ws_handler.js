document.addEventListener("DOMContentLoaded", () => {
    
    let ws;
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const exitBtn = document.getElementById("exitBtn");
    const resultadosDiv = document.getElementById("resultados");
    
    // --- ¡AÑADIDO! ---
    // Buscamos los selectores y el botón de config para deshabilitarlos
    const modelSelect = document.getElementById("model-select");
    const deviceSelect = document.getElementById("device-select");
    const computeSelect = document.getElementById("compute-select");
    const configBtn = document.querySelector("button[onclick='configurarModelo()']");
    const statusIndicator = document.getElementById("status-indicator");
    // --- FIN AÑADIDO ---

    const pInicial = resultadosDiv.querySelector("p");

    // Lógica de Debounce (esto está perfecto)
    let messageQueue = [];
    let debounceTimer = null;
    const DEBOUNCE_DELAY_MS = 150; 


    // Conecta el WebSocket al cargar la página
    connect();

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;
        ws = new WebSocket("ws://" + window.location.host + "/ws");

        ws.onopen = () => {
            console.log("✅ WebSocket conectado y esperando órdenes.");
            // --- ¡LÓGICA CORREGIDA! ---
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            if (configBtn) configBtn.disabled = false; // Asegurarse de que configBtn existe
            if (statusIndicator) statusIndicator.style.display = 'none';
            // --- FIN CORRECCIÓN ---
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const items = Array.isArray(data) ? data : [data];
                messageQueue.push(...items);
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    if (messageQueue.length > 0) {
                        actualizarResultadosEnHTML(messageQueue);
                        messageQueue = []; 
                    }
                }, DEBOUNCE_DELAY_MS);
            } catch (e) {
                console.error("Error procesando mensaje:", e);
            }
        };

        ws.onclose = () => {
            console.warn("🔌 WebSocket desconectado.");
            // --- ¡LÓGICA CORREGIDA! ---
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            if (configBtn) configBtn.disabled = false;
            if (statusIndicator) statusIndicator.style.display = 'none';
            // --- FIN CORRECCIÓN ---
        };

        ws.onerror = (error) => {
            console.error("❌ Error en WebSocket:", error);
            ws.close();
        };
    }

    // (Tu función 'actualizarResultadosEnHTML' está perfecta)
    function actualizarResultadosEnHTML(items) {
        if (pInicial && pInicial.style.display !== 'none') {
            pInicial.style.display = 'none';
        }
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
        resultadosDiv.prepend(fragment);
    }

    startBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'start'...");
            ws.send(JSON.stringify({ command: "start" }));
            
            // --- ¡LÓGICA CORREGIDA! ---
            startBtn.disabled = true;
            stopBtn.disabled = false;
            modelSelect.disabled = true;
            deviceSelect.disabled = true;
            computeSelect.disabled = true;
            if (configBtn) configBtn.disabled = true;
            if (statusIndicator) statusIndicator.style.display = 'flex';
            // --- FIN CORRECCIÓN ---
        }
    });

    stopBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'stop'...");
            ws.send(JSON.stringify({ command: "stop" }));

            // --- ¡LÓGICA CORREGIDA! ---
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            if (configBtn) configBtn.disabled = false;
            if (statusIndicator) statusIndicator.style.display = 'none';
            // --- FIN CORRECCIÓN ---
        }
    });

    exitBtn.addEventListener("click", async () => {
        if (ws) {
            ws.close();
        }
        await fetch("/exit", { method: "POST" });
    });
});