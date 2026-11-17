import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from dataset import make_loaders, CLASS_NAMES
from model import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def safe_macro_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")

def main():
    parser = argparse.ArgumentParser(description="Train MedScan ResNet-18 baseline")
    parser.add_argument("--data-dir", default="data/xray_raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    train_loader, val_loader, _, split_info = make_loaders(args.data_dir, args.batch_size, args.seed)
    model = build_model(3, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    best_auc = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train(); total_loss = 0.0; seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0); seen += labels.size(0)
        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.softmax(model(images.to(device)), dim=1)
                y_true.extend(labels.numpy().tolist()); y_prob.extend(probs.cpu().numpy().tolist())
        val_auc = safe_macro_auc(y_true, np.asarray(y_prob))
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), "val_macro_roc_auc": val_auc}
        history.append(row); print(json.dumps(row))
        score = val_auc if np.isfinite(val_auc) else -1.0
        if score > best_auc:
            best_auc = score
            torch.save({"state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                        "seed": args.seed, "split_info": split_info, "epoch": epoch + 1,
                        "val_macro_roc_auc": val_auc}, "models/best_resnet18.pt")
    Path("results/training_history.json").write_text(json.dumps(history, indent=2))
    Path("results/data_split_summary.json").write_text(json.dumps(split_info, indent=2))
    print(f"Training complete. Best validation macro ROC-AUC: {best_auc:.4f}")
    print("Next: python src/evaluate.py --checkpoint models/best_resnet18.pt")

if __name__ == "__main__":
    main()
