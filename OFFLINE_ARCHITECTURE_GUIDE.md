# Offline Operation & Edge Deployment Guide for ISL Translator

> **User Requirement**: *"We are making an API, we want to deploy it on the cloud, so if network is not available give me suggestions on what could make it work offline."*

---

## 1. Architectural Challenge & Solution Overview

Streaming 30 FPS video frames to a cloud server consumes high bandwidth (~2–5 Mbps upload) and requires stable, low-latency connectivity. If network connectivity drops:
- Frame streaming fails.
- Letter buffering stutters.
- Translation halts.

To solve this, we provide **three distinct offline deployment tiers**, plus a **Hybrid Auto-Fallback Architecture**.

```mermaid
graph TD
    A[Webcam Video Stream] --> B{Network Available?}
    B -- Yes (Cloud Active) --> C[Cloud FastAPI / WebSocket]
    C --> D[Cloud GPU / High-Accuracy Ensemble]
    B -- No (Offline) --> E[Client-Side ONNX Runtime Web / WebGPU]
    E --> F[In-Browser WASM Inference (Zero Latency)]
    D --> G[Letter Aggregator & Spellchecker]
    F --> G
    G --> H[Render Word & Speak Sentence]
```

---

## 2. Solution Tier 1: Progressive Web App (PWA) + In-Browser ONNX Runtime Web *(Recommended for Web Apps)*

This is the most elegant, seamless architecture for users accessing your web translator.

### How it works:
1. **Model Export**: We export `MobileNetV3-Small` or `YOLO11n` to ONNX format (quantized to INT8, shrinking file size down to **~2.5 MB**).
2. **Browser Caching via Service Worker**:
   - The browser downloads the web app and model file once.
   - A Service Worker caches `index.html`, `style.css`, `app.js`, and the `.onnx` weight file in IndexedDB / Cache Storage.
3. **Local Inference Execution**:
   - When the user loses internet connection, the browser runs the ONNX model locally using **ONNX Runtime Web** (`onnxruntime-web`) via WebAssembly (WASM) or WebGPU.
   - The webcam frames are processed entirely on the user's laptop/phone CPU/GPU.
   - **Data never leaves the device. Zero cloud dependency. 100% offline.**

### Implementation Snippet for Browser:
```javascript
// Load model directly in browser offline
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('/static/models/mobilenet_v3_small.onnx', {
    executionProviders: ['webgpu', 'wasm']
});

// Run local webcam frame inference
const inputTensor = new ort.Tensor('float32', preprocessedFrameData, [1, 3, 224, 224]);
const results = await session.run({ input: inputTensor });
```

---

## 3. Solution Tier 2: Local Edge Container (Docker Desktop / Edge Server)

Ideal for classrooms, hospitals, kiosks, or government offices where a dedicated PC or local server is present.

### How it works:
- Run the container locally using our provided `docker-compose.yml`:
  ```bash
  docker compose -f docker/docker-compose.yml up -d
  ```
- The entire FastAPI backend, temporal letter aggregator, and model run on `http://localhost:8000`.
- The user accesses the web app over `localhost` or the local Wi-Fi router (LAN) without needing an external internet connection.

---

## 4. Solution Tier 3: Native Mobile / IoT Edge (TFLite / ONNX Mobile)

Ideal if building an Android or iOS mobile application.

### How it works:
- Convert the trained PyTorch checkpoint to **TensorFlow Lite (`.tflite`)** or **ONNX Runtime Mobile**.
- Integrate with Flutter, React Native, or native Kotlin / Swift.
- Utilize on-device hardware accelerators:
  - **Android**: Android Neural Networks API (NNAPI) / Qualcomm Hexagon NPU.
  - **Apple iOS**: Apple Neural Engine (CoreML).
- Battery-efficient, real-time 30–60 FPS inference completely offline.

---

## 5. Recommended Hybrid Cloud-Edge Auto-Fallback Strategy

To get the best of both worlds:
1. **Online State**:
   - Client sends frames to Cloud FastAPI via WebSockets.
   - Cloud can run larger models, ensembled predictors, and LLM-based grammar correction.
2. **Heartbeat Monitoring**:
   - Client pings `/health` every 5 seconds.
3. **Automatic Fallback on Disconnect**:
   - If ping drops or latency exceeds 600ms, the frontend automatically switches inference from Cloud WebSocket to **local in-browser ONNX session**.
   - Displays a status chip: `⚡ Offline Mode (Local Edge Inference)`.
4. **Seamless Resumption**:
   - Once cloud connectivity is restored, the client reconnects to the cloud server without interrupting the user's current session.
