# Indian Sign Language (ISL) Real-Time Translator

An end-to-end, real-time Computer Vision and Deep Learning system that translates Indian Sign Language (ISL) gestures letter-by-letter, aggregates them into full words, performs predictive auto-completion, and structures complete English sentences with Text-to-Speech (TTS) audio output.

Includes a high-performance **FastAPI Cloud backend**, **Docker deployment**, a **zero-disk-extraction ZIP dataset loader**, model comparisons between **YOLO11n**, **MobileNetV3-Small**, and **EfficientNet-Lite0**, and an **offline edge deployment pipeline**.

---

## 📁 Project Folder Structure

```
d:/ISL Translator model/
├── data/
│   └── dataset.py             # PyTorch Dataset that loads dataset_ISL.zip directly in-memory (0 disk extraction)
├── docker/
│   ├── Dockerfile             # Multi-stage production container with OpenCV & PyTorch
│   ├── docker-compose.yml     # Container orchestration with volume mounts & env configs
│   └── .dockerignore          # Docker build exclusion rules
├── models/
│   ├── checkpoints/           # Trained PyTorch model weights (.pth)
│   ├── onnx/                  # Exported ONNX format models for offline edge/browser inference
│   └── README.md              # Architecture trade-offs and model comparison notes
├── notebooks/
│   └── ISL_Model_Exploration_and_Comparison.ipynb # EDA, zero-disk sampling, benchmarking, & ONNX export
├── src/
│   ├── api/
│   │   └── server.py          # FastAPI application with REST & real-time WebSockets
│   ├── core/
│   │   ├── aggregator.py      # Letter debounce, word formation, autocomplete, & sentence builder
│   │   ├── infer.py           # Unified inference engine (PyTorch & ONNX Runtime)
│   │   ├── train.py           # Training pipeline directly reading from ZIP archive
│   │   └── compare.py         # Multi-model benchmarking script (latency, FPS, size, accuracy)
│   ├── models/
│   │   ├── factory.py         # Model factory: MobileNetV3-Small, EfficientNet-Lite0, YOLO11n
│   │   └── export_onnx.py     # ONNX model exporter for edge & offline execution
│   └── main.py                # Server runner entrypoint
├── static/
│   ├── index.html             # Sleek dark-mode web dashboard with webcam overlay
│   ├── style.css              # Custom styling with glassmorphism & live meters
│   └── app.js                 # WebSocket client, stability meter, TTS speaker, & fallback logic
├── dataset_ISL.zip            # Original dataset archive (12,637 images, 26 alphabet classes)
├── requirements.txt           # Python dependency specifications
├── .env                       # Active environment configurations
├── .env.example               # Template environment configurations
├── OFFLINE_ARCHITECTURE_GUIDE.md # Detailed offline deployment blueprint
└── README.md                  # Project documentation
```

---

## ⚡ Zero-Disk Extraction Dataset Loading

As required, **the 120MB dataset archive (`dataset_ISL.zip`) is read directly into RAM without extracting files onto your hard drive**:
- Implemented in [`data/dataset.py`](file:///d:/ISL%20Translator%20model/data/dataset.py).
- Uses `zipfile.ZipFile` with lazy per-worker handles for thread-safe PyTorch `DataLoader` operation.
- Reads image bytes on-the-fly via `io.BytesIO` and decodes directly with PIL.
- Generates stratified Train/Val/Test splits purely from archive index metadata without writing a single temporary file.

---

## 🤖 Model Architectures & Comparative Evaluation

The project implements and benchmarks the three requested architectures for 26 alphabet classes (`a` through `z`):

| Model Candidate | Parameter Count | FP32 Size | Est. INT8 Quantized Size | Inference Latency (CPU) | Primary Strengths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** | **~2.5 M** | **~9.8 MB** | **~2.5 MB** | **~12–18 ms** | Highest FPS on low-power CPUs & browser WebAssembly |
| **EfficientNet-Lite0** | **~4.7 M** | **~18.5 MB** | **~4.8 MB** | **~24–32 ms** | Quantization-friendly, high accuracy-to-FLOP ratio |
| **YOLO11n-cls** | **~1.6–2.8 M** | **~6.5 MB** | **~1.8 MB** | **~15–22 ms** | Modern feature representations & fast GPU/NPU throughput |

> **Clarification on YOLO Versions**: Ultralytics officially launched **YOLO11** in late 2024. "YOLO26n" does not exist (likely a community typo for YOLO11 or YOLOv8). We have implemented **YOLO11n** as the nano YOLO classifier candidate.

To run the automated model benchmark comparison across all three candidates:
```bash
python -m src.core.compare --zip_path dataset_ISL.zip --out_dir models
```
This generates:
- `models/model_comparison_metrics.json`
- `models/model_comparison_table.csv`
- `models/model_comparison_charts.png` (4-panel visual dashboard)

---

## 🧠 Letter-by-Letter Aggregation & Sentence Formation

Raw frame-by-frame predictions fluctuate rapidly. The custom [`SignAggregator`](file:///d:/ISL%20Translator%20model/src/core/aggregator.py) engine solves this:
1. **Confidence Thresholding**: Predictions below confidence threshold (e.g. 70%) are filtered out.
2. **Temporal Consistency (Debounce)**: A sign must be held steady for `N` consecutive frames (e.g., 8 frames) before confirming the letter.
3. **Word Buffering & Auto-Completion**: Confirmed letters accumulate into words. Top-5 English completions are dynamically suggested.
4. **Auto-Space & Spell-Correction**: Inactivity pauses (> 2.0s) trigger word commit and automatic spelling correction.
5. **Sentence Assembly & TTS**: Assembled sentences are formatted with punctuation and can be spoken aloud via browser speech synthesis.

---

## 🌐 Cloud API & Real-Time Webcam Web App

### Run Locally:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI Server
python src/main.py
```
- Web Application: **http://localhost:8000**
- Interactive Swagger API Docs: **http://localhost:8000/docs**

### Run via Docker:
```bash
docker compose -f docker/docker-compose.yml up --build -d
```

---

## 📶 Offline Operation (When Cloud Network is Unavailable)

If internet connectivity drops, you have three production strategies (detailed in [`OFFLINE_ARCHITECTURE_GUIDE.md`](file:///d:/ISL%20Translator%20model/OFFLINE_ARCHITECTURE_GUIDE.md)):

1. **Client-Side Progressive Web App (PWA) + In-Browser ONNX Runtime Web**:
   - Model is exported to ONNX via `python -m src.models.export_onnx --model mobilenet_v3_small`.
   - The browser caches the web page and ONNX file.
   - Using WebAssembly / WebGPU, the client device executes the neural network locally without sending a single frame across the internet.
2. **Local Edge Docker Deployment**:
   - Run `docker compose -f docker/docker-compose.yml up -d` directly on the local machine or edge box.
3. **Hybrid Auto-Fallback**:
   - The web app automatically switches to local in-browser inference if cloud ping fails.

---

## 🏋️ Model Training

To train any model directly from the ZIP file:
```bash
python -m src.core.train --zip_path dataset_ISL.zip --model mobilenet_v3_small --epochs 15 --batch_size 32
```
Checkpoints will be saved automatically to `models/checkpoints/`.
