# ISL Models Directory

This folder holds trained neural network checkpoints, exported ONNX models, and evaluation comparison benchmarks.

## Subdirectories:
- `checkpoints/`: PyTorch model weights (`.pth`) for `mobilenet_v3_small_best.pth`, `efficientnet_lite0_best.pth`, and `yolo11n_best.pth`.
- `onnx/`: Exported ONNX format models (`.onnx`) optimized for edge offline inference with ONNX Runtime / WebAssembly.

## Candidate Model Comparison Summary:

| Architecture | Parameters | FP32 Size | Est. INT8 Quantized Size | Target Deployment Profile |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** | ~2.5M | ~9.8 MB | ~2.5 MB | **Best for Browser / Edge CPU / Offline PWA** |
| **EfficientNet-Lite0** | ~4.7M | ~18.5 MB | ~4.8 MB | **Best for High Accuracy / Low FLOPs on Mobile** |
| **YOLO11n-cls** | ~1.6M | ~6.5 MB | ~1.8 MB | **Best for Modern Vision Backbone / Fast GPU & NPU** |

> *Note on YOLO26n*: Ultralytics official versions currently culminate in **YOLO11** (released late 2024). YOLO26n is a non-existent or community typo. YOLO11n is the state-of-the-art nano candidate implemented in this project.
