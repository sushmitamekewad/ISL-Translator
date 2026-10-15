import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
try:
    from src.models.factory import create_model, count_parameters
except ModuleNotFoundError:
    from models.factory import create_model, count_parameters

from data.dataset import build_dataloaders

def benchmark_latency(model: nn.Module, device: str, input_shape=(1, 3, 224, 224), num_warmup: int = 20, num_runs: int = 100) -> tuple:
    """Accurately measures inference latency and throughput (FPS)."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    if device == 'cuda':
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    return avg_ms, std_ms, fps


def evaluate_accuracy(model: nn.Module, test_loader, device: str, max_batches: int = 20) -> tuple:
    """Evaluates Top-1 and Top-5 accuracy on a subset or full test set."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if max_batches and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1
            _, preds = torch.max(outputs, 1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

            total += labels.size(0)

    acc1 = (correct_top1 / total) * 100 if total > 0 else 0.0
    acc5 = (correct_top5 / total) * 100 if total > 0 else 0.0
    return acc1, acc5


def run_model_comparison(zip_path: str = "dataset_ISL.zip", output_dir: str = "models", max_eval_batches: int = None, auto_train: bool = True, train_epochs: int = 10):
    """
    Compares all three ISL model candidates using TRAINED checkpoints.
    If a trained checkpoint is not found and auto_train=True, the model is
    automatically fine-tuned before benchmarking.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = "cpu"
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            device = "cuda"
        except Exception:
            device = "cpu"
    print(f"=== ISL Model Benchmark & Comparison Pipeline ===")
    print(f"Hardware Device: {device.upper()}")
    print(f"Auto-train missing models: {auto_train}")

    candidate_models = ["mobilenet_v3_small", "efficientnet_lite0", "yolo11n"]

    # Load data splits directly from ZIP
    train_loader, val_loader, test_loader, classes = build_dataloaders(zip_path=zip_path, batch_size=32)
    num_classes = len(classes)
    print(f"Dataset loaded: {num_classes} classes, test batches: {len(test_loader)}")

    results = []

    for model_name in candidate_models:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {model_name}")
        print(f"{'='*60}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        has_checkpoint = os.path.exists(checkpoint_path)

        # --- Step 1: Load or Train the model ---
        if has_checkpoint:
            print(f"  [OK] Found trained checkpoint: {checkpoint_path}")
            model = create_model(model_name, num_classes=num_classes, pretrained=False)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state.get("state_dict", state))
            model.to(device)
            print(f"  [OK] Loaded trained weights successfully.")
        elif auto_train:
            print(f"  [!] No trained checkpoint found. Auto-training {model_name} for {train_epochs} epochs...")
            model = _train_model_for_comparison(
                model_name=model_name,
                num_classes=num_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                device=device,
                epochs=train_epochs
            )
            print(f"  [OK] Training completed. Checkpoint saved to: {checkpoint_path}")
        else:
            print(f"  [SKIP] No checkpoint found and auto_train=False. Skipping {model_name}.")
            continue

        model.eval()

        # --- Step 2: Parameter count & footprint ---
        param_count, size_mb = count_parameters(model)

        # --- Step 3: Latency & FPS ---
        avg_ms, std_ms, fps = benchmark_latency(model, device=device)

        # --- Step 4: Accuracy on test split (using TRAINED model) ---
        top1, top5 = evaluate_accuracy(model, test_loader, device=device, max_batches=max_eval_batches)

        metric = {
            "model": model_name,
            "params_millions": round(param_count / 1e6, 2),
            "size_fp32_mb": round(size_mb, 2),
            "size_int8_mb": round(size_mb / 4, 2),
            "latency_ms": round(avg_ms, 2),
            "latency_std_ms": round(std_ms, 2),
            "throughput_fps": round(fps, 1),
            "test_top1_acc_pct": round(top1, 2),
            "test_top5_acc_pct": round(top5, 2),
            "edge_onnx_ready": True,
            "trained": True
        }
        results.append(metric)
        print(f"  Results: Params={metric['params_millions']}M | Size={metric['size_fp32_mb']}MB | Latency={metric['latency_ms']}ms ({metric['throughput_fps']} FPS) | Top-1={metric['test_top1_acc_pct']}% | Top-5={metric['test_top5_acc_pct']}%")

        # Clean up GPU memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    if not results:
        print("No models were benchmarked. Train at least one model first.")
        return None

    # Save to JSON
    json_path = os.path.join(output_dir, "model_comparison_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "model_comparison_table.csv")
    df.to_csv(csv_path, index=False)

    # Generate visual comparison charts
    plot_comparison_charts(df, os.path.join(output_dir, "model_comparison_charts.png"))
    print(f"\n{'='*60}")
    print(f"All comparison artifacts generated at: {output_dir}/")
    print(f"{'='*60}")
    return df


def _train_model_for_comparison(
    model_name: str,
    num_classes: int,
    train_loader,
    val_loader,
    checkpoint_path: str,
    device: str,
    epochs: int = 10,
    lr: float = 0.001
) -> nn.Module:
    """Internal helper to auto-train a model during comparison if no checkpoint exists."""
    from tqdm import tqdm

    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"  [{model_name}] Epoch {epoch}/{epochs} Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  [{model_name}] Epoch {epoch}/{epochs} Val", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"  [{model_name}] Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": [chr(ord('a') + i) for i in range(num_classes)]
            }, checkpoint_path)
            print(f"  >>> New best checkpoint saved (Val Acc: {best_val_acc*100:.2f}%)")

    # Reload best checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    model.to(device)
    model.eval()
    return model


def plot_comparison_charts(df: pd.DataFrame, save_path: str):
    """Creates a 4-panel dashboard comparing model metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Indian Sign Language Model Comparison: YOLO11n vs MobileNetV3-Small vs EfficientNet-Lite0", fontsize=15, fontweight='bold')

    colors = ['#3b82f6', '#10b981', '#f59e0b']

    # 1. Model Size vs Parameters
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['model'], df['size_fp32_mb'], color=colors, width=0.5)
    ax1.set_title("Model Footprint (FP32 Size in MB)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Size (MB)")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f} MB", ha='center', va='bottom', fontweight='bold')

    # 2. Inference Latency (ms)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model'], df['latency_ms'], color=colors, width=0.5)
    ax2.set_title("Inference Latency per Frame (ms - Lower is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Latency (ms)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')

    # 3. Throughput (FPS)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['model'], df['throughput_fps'], color=colors, width=0.5)
    ax3.set_title("Throughput (Frames Per Second - Higher is Better)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("FPS")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.0f} FPS", ha='center', va='bottom', fontweight='bold')

    # 4. Latency vs Accuracy Trade-off
    ax4 = axes[1, 1]
    for i, row in df.iterrows():
        ax4.scatter(row['latency_ms'], row['test_top1_acc_pct'], color=colors[i], s=200, label=row['model'])
        ax4.annotate(row['model'], (row['latency_ms'] + 0.3, row['test_top1_acc_pct']), fontweight='bold')
    ax4.set_title("Latency vs Accuracy Trade-off", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Latency (ms)")
    ax4.set_ylabel("Top-1 Accuracy (%)")
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ISL models (trains missing models automatically)")
    parser.add_argument("--zip_path", type=str, default="dataset_ISL.zip")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--batches", type=int, default=None, help="Max test batches (None=full test set)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for auto-trained models")
    parser.add_argument("--no_auto_train", action="store_true", help="Skip models without existing checkpoints")
    args = parser.parse_args()

    run_model_comparison(
        zip_path=args.zip_path,
        output_dir=args.out_dir,
        max_eval_batches=args.batches,
        auto_train=not args.no_auto_train,
        train_epochs=args.epochs
    )

