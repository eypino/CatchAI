// Esta función es llamada por el 'onclick' del botón en el HTML.
async function configurarModelo() {
    const modelSelect = document.getElementById("model-select");
    const deviceSelect = document.getElementById("device-select");
    const computeSelect = document.getElementById("compute-select");

    const config = {
        model_size: modelSelect.value,
        device: deviceSelect.value,
        compute_type: computeSelect.value,
    };

    try {
        const response = await fetch("/configurar_modelo", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(config),
        });
        const result = await response.json();
        console.log("Configuración aplicada:", result);
        alert(`Modelo configurado a: ${result.modelo} en ${result.device}`);
    } catch (error) {
        console.error("Error al configurar el modelo:", error);
        alert("Error al aplicar la configuración.");
    }
}