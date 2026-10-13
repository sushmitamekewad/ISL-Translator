// Indian Sign Language (ISL) Real-Time Client Application
document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("captureCanvas");
    const ctx = canvas.getContext("2d");
    
    // UI Elements
    const toggleCameraBtn = document.getElementById("toggleCameraBtn");
    const detectedLetter = document.getElementById("detectedLetter");
    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceVal = document.getElementById("confidenceVal");
    const stabilityFill = document.getElementById("stabilityFill");
    const currentWordDisplay = document.getElementById("currentWordDisplay");
    const suggestionsContainer = document.getElementById("suggestionsContainer");
    const sentenceText = document.getElementById("sentenceText");
    const top3Container = document.getElementById("top3Container");
    const fpsCounter = document.getElementById("fpsCounter");
    const latencyCounter = document.getElementById("latencyCounter");
    const modelSelect = document.getElementById("modelSelect");
    const networkBadge = document.getElementById("networkBadge");
    const networkStatusText = document.getElementById("networkStatusText");
    
    // Action Buttons
    const speakBtn = document.getElementById("speakBtn");
    const copyBtn = document.getElementById("copyBtn");
    const spaceBtn = document.getElementById("spaceBtn");
    const backspaceBtn = document.getElementById("backspaceBtn");
    const clearAllBtn = document.getElementById("clearAllBtn");

    // State Variables
    let isStreaming = false;
    let streamInterval = null;
    let ws = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let currentFps = 0;
    let isProcessing = false;

    // Initialize Canvas Size
    canvas.width = 224;
    canvas.height = 224;

    // Connect WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        
        try {
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log("[WebSocket] Connected to translation server.");
                updateNetworkStatus(true, "Cloud WebSocket Live");
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleServerResponse(data.letter, data.confidence, data.latency_ms, data.top_3, data.state);
                isProcessing = false;
            };

            ws.onerror = (err) => {
                console.warn("[WebSocket] Socket error, falling back to HTTP REST", err);
                updateNetworkStatus(false, "HTTP REST Mode");
            };

            ws.onclose = () => {
                console.log("[WebSocket] Closed.");
                updateNetworkStatus(false, "Reconnecting...");
                setTimeout(connectWebSocket, 3000);
            };
        } catch (e) {
            console.warn("[WebSocket] Setup failed, will use HTTP POST:", e);
        }
    }

    function updateNetworkStatus(online, text) {
        if (online) {
            networkBadge.querySelector(".status-dot").className = "status-dot online";
            networkStatusText.textContent = text || "Online";
        } else {
            networkBadge.querySelector(".status-dot").className = "status-dot";
            networkBadge.querySelector(".status-dot").style.backgroundColor = "#f59e0b";
            networkStatusText.textContent = text || "Offline / Fallback";
        }
    }

    // Camera Toggle
    toggleCameraBtn.addEventListener("click", async () => {
        if (!isStreaming) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: "user" }
                });
                video.srcObject = stream;
                isStreaming = true;
                toggleCameraBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Webcam';
                toggleCameraBtn.classList.replace("btn-primary", "btn-secondary");

                // Start capture loop (~15-20 FPS for optimal inference without lag)
                streamInterval = setInterval(captureAndSendFrame, 65);
            } catch (err) {
                alert("Could not access webcam: " + err.message);
            }
        } else {
            stopCamera();
        }
    });

    function stopCamera() {
        if (streamInterval) clearInterval(streamInterval);
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        isStreaming = false;
        toggleCameraBtn.innerHTML = '<span class="btn-icon">📹</span> Start Webcam';
        toggleCameraBtn.classList.replace("btn-secondary", "btn-primary");
        detectedLetter.textContent = "--";
        confidenceVal.textContent = "Waiting...";
        confidenceBar.style.width = "0%";
        stabilityFill.style.width = "0%";
    }

    // Frame Capture & Dispatch
    function captureAndSendFrame() {
        if (!isStreaming || isProcessing) return;

        // Draw centered square crop of webcam to canvas (224x224)
        const vW = video.videoWidth;
        const vH = video.videoHeight;
        if (!vW || !vH) return;

        const size = Math.min(vW, vH);
        const startX = (vW - size) / 2;
        const startY = (vH - size) / 2;

        ctx.drawImage(video, startX, startY, size, size, 0, 0, 224, 224);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

        // Calculate FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            currentFps = frameCount;
            frameCount = 0;
            lastFpsTime = now;
            fpsCounter.textContent = `${currentFps} FPS`;
        }

        isProcessing = true;

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ frame: dataUrl }));
        } else {
            // HTTP POST fallback
            fetch("/stream/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: dataUrl })
            })
            .then(res => res.json())
            .then(data => {
                handleServerResponse(
                    data.inference.predicted_letter,
                    data.inference.confidence,
                    data.inference.latency_ms,
                    data.inference.top_3,
                    data.aggregation
                );
            })
            .catch(err => console.error("Stream POST error:", err))
            .finally(() => {
                isProcessing = false;
            });
        }
    }

    // Process & Render Translation Updates
    function handleServerResponse(letter, confidence, latencyMs, top3, state) {
        latencyCounter.textContent = `${latencyMs} ms`;

        // Render Active Letter & Confidence
        detectedLetter.textContent = letter || "--";
        const pct = Math.round(confidence * 100);
        confidenceVal.textContent = `${pct}%`;
        confidenceBar.style.width = `${pct}%`;

        // Stability Debounce Meter
        const stabilityPct = Math.round((state.stability_ratio || 0) * 100);
        stabilityFill.style.width = `${stabilityPct}%`;

        // Render Current Word
        if (state.current_word) {
            currentWordDisplay.textContent = state.current_word;
        } else {
            currentWordDisplay.innerHTML = '<span class="word-placeholder">Signs will accumulate here...</span>';
        }

        // Render Suggestions
        renderSuggestions(state.suggestions || []);

        // Render Sentence
        if (state.full_sentence) {
            sentenceText.textContent = state.full_sentence;
        } else {
            sentenceText.textContent = "No words translated yet.";
        }

        // Render Top 3 Probabilities
        if (top3 && top3.length > 0) {
            top3Container.innerHTML = top3.map(item => `
                <div class="prob-row">
                    <span class="prob-letter">${item.label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${Math.round(item.confidence * 100)}%"></div>
                    </div>
                    <span class="prob-num">${Math.round(item.confidence * 100)}%</span>
                </div>
            `).join("");
        }
    }

    function renderSuggestions(suggestions) {
        if (!suggestions || suggestions.length === 0) {
            suggestionsContainer.innerHTML = '<span class="empty-hint">No suggestions yet</span>';
            return;
        }

        suggestionsContainer.innerHTML = suggestions.map(word => `
            <button class="suggestion-chip" data-word="${word}">${word}</button>
        `).join("");

        suggestionsContainer.querySelectorAll(".suggestion-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                sendAggregatorAction("select_suggestion", chip.getAttribute("data-word"));
            });
        });
    }

    // Send Aggregator Actions
    async function sendAggregatorAction(action, value = null) {
        try {
            const res = await fetch("/aggregator/action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action, value })
            });
            const data = await res.json();
            
            if (data.current_word) {
                currentWordDisplay.textContent = data.current_word;
            } else {
                currentWordDisplay.innerHTML = '<span class="word-placeholder">Signs will accumulate here...</span>';
            }
            sentenceText.textContent = data.full_sentence || "No words translated yet.";
            renderSuggestions(data.suggestions || []);
        } catch (e) {
            console.error("Action failed:", e);
        }
    }

    // Controls
    backspaceBtn.addEventListener("click", () => sendAggregatorAction("backspace"));
    spaceBtn.addEventListener("click", () => sendAggregatorAction("space"));
    clearAllBtn.addEventListener("click", () => sendAggregatorAction("clear"));

    // Text to Speech
    speakBtn.addEventListener("click", () => {
        const text = sentenceText.textContent;
        if (!text || text === "No words translated yet.") return;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    });

    // Copy to Clipboard
    copyBtn.addEventListener("click", () => {
        const text = sentenceText.textContent;
        if (!text || text === "No words translated yet.") return;
        navigator.clipboard.writeText(text).then(() => {
            const original = copyBtn.innerHTML;
            copyBtn.innerHTML = "<span>✓</span> Copied!";
            setTimeout(() => { copyBtn.innerHTML = original; }, 1500);
        });
    });

    // Model Selector Switch
    modelSelect.addEventListener("change", async (e) => {
        const selectedModel = e.target.value;
        try {
            const res = await fetch("/models/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: selectedModel })
            });
            const data = await res.json();
            console.log("Model switched:", data.message);
        } catch (err) {
            console.error("Failed to switch model:", err);
        }
    });

    // Initialize WebSocket
    connectWebSocket();
});
