document.addEventListener("DOMContentLoaded", () => {   
    // ================== ¡NUEVO! LÓGICA DE SMOOTH SCROLL ==================
    // Busca todos los enlaces que empiezan con '#'
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault(); // Prevenir el salto brusco
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    // ================== FIN LÓGICA DE SCROLL ==================


    // --- Lógica del Panel Deslizante (Esta sí la necesitamos) ---
    const settingsBtn = document.getElementById("settingsBtn");
    const closeSettingsBtn = document.getElementById("closeSettingsBtn");
    const settingsPanel = document.getElementById("settingsPanel");
    const settingsOverlay = document.getElementById("settingsOverlay");

    const togglePanel = (open) => {
        if (!settingsPanel || !settingsOverlay) return; // Seguridad
        if (open) {
            settingsPanel.classList.remove("translate-x-full");
            settingsOverlay.classList.add("visible");
        } else {
            settingsPanel.classList.add("translate-x-full");
            settingsOverlay.classList.remove("visible");
        }
    };

    if (settingsBtn) settingsBtn.addEventListener("click", () => togglePanel(true));
    if (closeSettingsBtn) closeSettingsBtn.addEventListener("click", () => togglePanel(false));
    if (settingsOverlay) settingsOverlay.addEventListener("click", () => togglePanel(false));

    // --- Lógica del Modo Oscuro/Claro (Esta la necesitamos) ---
    const darkModeToggle = document.getElementById("darkModeToggle");
    
    const applyTheme = (isDark) => {
        const body = document.body;
        const html = document.documentElement;

        if (isDark) {
            html.classList.add("dark");
            body.classList.add("bg-background-dark");
            body.classList.remove("bg-background-light");
            if (darkModeToggle) darkModeToggle.checked = true;
        } else {
            html.classList.remove("dark");
            body.classList.remove("bg-background-dark");
            body.classList.add("bg-background-light");
            if (darkModeToggle) darkModeToggle.checked = false;
        }
    };

    const savedTheme = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    
    if (savedTheme === "dark" || (!savedTheme && prefersDark)) {
        applyTheme(true);
    } else {
        applyTheme(false);
    }

    if (darkModeToggle) {
        darkModeToggle.addEventListener("change", () => {
            if (darkModeToggle.checked) {
                applyTheme(true);
                localStorage.setItem("theme", "dark");
            } else {
                applyTheme(false);
                localStorage.setItem("theme", "light");
            }
        });
    }
});