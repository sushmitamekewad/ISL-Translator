import sys
import os
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
try:
    from src.models.factory import create_model
except ModuleNotFoundError:
    from models.factory import create_model

def export_model_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    dynamic_batch: bool = True
) -> str:
    """
    Exports a PyTorch model to ONNX format.
    This ONNX model can run completely offline via:
      1. onnxruntime on Python / edge servers
      2. ONNX Runtime Web (WebAssembly / WebGPU) in browser clients
      3. Mobile apps (React Native, Flutter, Swift, Kotlin)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape)

    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None

    print(f"[ONNX Exporter] Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX Exporter] Successfully exported ONNX model ({file_size_mb:.2f} MB)")

    # Validate ONNX model with onnx and onnxruntime if available
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX Exporter] ONNX model structure validated successfully.")
    except Exception as e:
        print(f"[ONNX Exporter] Note: Skipping onnx verification check: {e}")

    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        test_in = np.random.randn(*input_shape).astype(np.float32)
        outs = session.run(None, {'input': test_in})
        print(f"[ONNX Exporter] Test inference with onnxruntime succeeded. Output shape: {outs[0].shape}")
    except Exception as e:
        print(f"[ONNX Exporter] Note: onnxruntime test run skipped: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ISL model to ONNX")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_lite0", "yolo11n"])
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model = create_model(args.model, num_classes=26, pretrained=(args.weights is None))
    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        print(f"Loaded weights from {args.weights}")

    out_path = args.out or f"models/onnx/{args.model}.onnx"
    export_model_to_onnx(model, out_path)
