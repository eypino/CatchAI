document.addEventListener("DOMContentLoaded", () => {
    const videoElement = document.getElementById('avatar-stream');
    const statusElement = document.getElementById('stream-status');

    if (!videoElement) {
        console.error("No se encontró el elemento de video #avatar-stream.");
        return;
    }

    // Esta función pide acceso a la cámara.
    async function startVirtualCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            videoElement.srcObject = stream;
            videoElement.play();
            
            statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i>Cámara virtual activa';
            console.log("✅ Cámara virtual conectada.");

        } catch (error) {
            console.error("❌ Error al acceder a la cámara virtual:", error);
            statusElement.innerHTML = '<i class="fas fa-circle text-danger me-1"></i>Error de cámara';
            alert("No se pudo acceder a la cámara virtual.\n\nAsegúrate de haber iniciado la 'Cámara Virtual' en OBS y de haber dado permiso al navegador.");
        }
    }

    // Iniciar el proceso.
    startVirtualCamera();
});