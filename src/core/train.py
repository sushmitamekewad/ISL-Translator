import sys
import os
import argparse
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import build_dataloaders
try:
    from src.models.factory import create_model
except ModuleNotFoundError:
    from models.factory import create_model

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> tuple:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train ISL Model directly from ZIP file")
    parser.add_argument("--zip_path", type=str, default="dataset_ISL.zip", help="Path to dataset ZIP file")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_lite0", "yolo11n"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--from_scratch", action="store_true", help="Train from random initialization without pre-trained weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting ISL Training for {args.model} on {device} ===")
    print(f"Loading data directly from archive: {args.zip_path} (Zero disk extraction)")
    print(f"Mode: {'From Scratch (Random Initialization)' if args.from_scratch else 'Transfer Learning (Pre-trained Backbone)'}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        zip_path=args.zip_path,
        batch_size=args.batch_size
    )
    print(f"Dataset loaded. Classes: {len(classes)} ({classes[:5]} ... {classes[-5:]})")

    model = create_model(args.model, num_classes=len(classes), pretrained=not args.from_scratch)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.save_dir, f"{args.model}_best.pth")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes
            }, best_checkpoint_path)
            print(f"  >>> Saved new best model to {best_checkpoint_path} (Val Acc: {best_val_acc*100:.2f}%)")

    print(f"\nTraining completed! Best checkpoint: {best_checkpoint_path}")

if __name__ == "__main__":
    main()
