import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess

MODELS = ["mobilenet_v3_small", "efficientnet_lite0", "yolo11n"]

def run_training_for_all(epochs: int = 5, batch_size: int = 32, zip_path: str = "dataset_ISL.zip", lr: float = 0.001):
    python_exe = sys.executable

    print("=================================================================")
    print("      Starting Training Pipeline for All 3 ISL Models            ")
    print(f"      Models: {', '.join(MODELS)}                                ")
    print(f"      Epochs per model: {epochs} | Batch size: {batch_size}       ")
    print("=================================================================\n")

    for idx, model_name in enumerate(MODELS, 1):
        print(f"\n[{idx}/3] >>> Starting Training for: {model_name} ...")
        cmd = [
            python_exe,
            "-m", "src.core.train",
            "--model", model_name,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--zip_path", zip_path,
            "--lr", str(lr)
        ]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print(f"[ERROR] Training failed for model {model_name} with returncode {res.returncode}")
        else:
            print(f"[SUCCESS] Completed training for {model_name}!")
            
            # Export to ONNX after successful training
            weights_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", f"{model_name}_best.pth")
            if os.path.exists(weights_path):
                print(f"[*] Exporting {model_name} to ONNX...")
                export_cmd = [
                    python_exe,
                    "-m", "src.models.export_onnx",
                    "--model", model_name,
                    "--weights", weights_path
                ]
                subprocess.run(export_cmd, cwd=PROJECT_ROOT)

    print("\n=================================================================")
    print(" [*] Running Comparative Evaluation & Benchmark Across All 3 ...")
    print("=================================================================")
    compare_cmd = [
        python_exe,
        "-m", "src.core.compare",
        "--zip_path", zip_path,
        "--out_dir", "models"
    ]
    subprocess.run(compare_cmd, cwd=PROJECT_ROOT)

    print("\n[COMPLETE] All 3 models have been trained, exported to ONNX, and benchmarked!")
    print("Checkpoints saved in: models/checkpoints/")
    print("ONNX models saved in: models/onnx/")
    print("Benchmark charts and metrics saved in: models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all 3 ISL models sequentially")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per model (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--zip_path", type=str, default="dataset_ISL.zip", help="Path to dataset ZIP file")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    run_training_for_all(epochs=args.epochs, batch_size=args.batch_size, zip_path=args.zip_path, lr=args.lr)
