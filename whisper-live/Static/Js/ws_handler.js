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
    
    // --- ¡NUEVO! Elementos de subida de video ---
    const uploadBtn = document.getElementById("upload-video-btn");
    const uploadInput = document.getElementById("video-upload-input");
    const uploadStatus = document.getElementById("upload-status");
    const closeSettingsBtn = document.getElementById("closeSettingsBtn");
    // --- FIN NUEVO ---

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
            setUIState('stopped'); // Usar función helper
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const items = Array.isArray(data) ? data : [data];
                
                // --- ¡NUEVO! Manejar auto-stop ---
                // El broadcast loop envía esto cuando el archivo termina
                const stopMessage = items.find(item => item.status === 'stopped');
                if (stopMessage) {
                    console.log("Recibido comando 'stopped' (fin de archivo).");
                    setUIState('stopped');
                    if (uploadStatus) uploadStatus.textContent = "Procesamiento de archivo finalizado.";
                    return; // No es una transcripción
                }
                // --- FIN NUEVO ---
                
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
            setUIState('stopped'); // Estado detenido
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

    // --- ¡NUEVO! Función para cambiar estado de UI ---
    function setUIState(state) {
        if (state === 'running') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            modelSelect.disabled = true;
            deviceSelect.disabled = true;
            computeSelect.disabled = true;
            if (configBtn) configBtn.disabled = true;
            if (uploadBtn) uploadBtn.disabled = true; // Deshabilitar subida
            if (statusIndicator) statusIndicator.style.display = 'flex';
        } else { // 'stopped'
            startBtn.disabled = false;
            stopBtn.disabled = true;
            modelSelect.disabled = false;
            deviceSelect.disabled = false;
            computeSelect.disabled = false;
            if (configBtn) configBtn.disabled = false;
            if (uploadBtn) uploadBtn.disabled = false; // Habilitar subida
            if (uploadStatus) uploadStatus.textContent = ""; // Limpiar estado
            if (statusIndicator) statusIndicator.style.display = 'none';
        }
    }
    // --- FIN NUEVO ---


    startBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'start' (en vivo)...");
            ws.send(JSON.stringify({ command: "start" }));
            setUIState('running'); // Usar función helper
        }
    });

    stopBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Enviando comando 'stop'...");
            ws.send(JSON.stringify({ command: "stop" }));
            setUIState('stopped'); // Usar función helper
        }
    });
    
    // --- ¡NUEVO! Event listener para el botón de subida ---
    if (uploadBtn) {
        uploadBtn.addEventListener("click", async () => {
            if (!uploadInput || !uploadInput.files[0]) {
                if (uploadStatus) uploadStatus.textContent = "Por favor, selecciona un video.";
                return;
            }
            
            const file = uploadInput.files[0];
            const formData = new FormData();
            formData.append("file", file);

            setUIState('running'); // Poner UI en modo "cargando"
            if (uploadStatus) uploadStatus.textContent = "Subiendo y procesando...";
            if (closeSettingsBtn) closeSettingsBtn.click(); // Cerrar panel

            try {
                const response = await fetch("/upload_video", {
                    method: "POST",
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.status === "ok") {
                    console.log("Subida exitosa. Procesamiento iniciado.");
                    if (uploadStatus) uploadStatus.textContent = "Procesando, espera...";
                    // La UI ya está en modo 'running'
                } else {
                    console.error("Error en la subida:", data.message);
                    if (uploadStatus) uploadStatus.textContent = "Error: " + data.message;
                    setUIState('stopped'); // Hubo un error, volver a 'stopped'
                }
            } catch (error) {
                console.error("Error de red al subir:", error);
                if (uploadStatus) uploadStatus.textContent = "Error de red.";
                setUIState('stopped');
            }
        });
    }
    // --- FIN NUEVO ---

    exitBtn.addEventListener("click", async () => {
        if (ws) {
            ws.close();
        }
        await fetch("/exit", { method: "POST" });
    });
});