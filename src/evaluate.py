<<<<<<< HEAD
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from dataset import make_loaders, CLASS_NAMES
from model import build_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedScan baseline on held-out test split")
    parser.add_argument("--data-dir", default="data/xray_raw")
    parser.add_argument("--checkpoint", default="models/best_resnet18.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, split_info = make_loaders(args.data_dir, args.batch_size, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(3, pretrained=False).to(device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state); model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(probabilities.argmax(1).cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())
    y_prob = np.asarray(y_prob); cm = confusion_matrix(y_true, y_pred, labels=list(range(3)))
    report = classification_report(y_true, y_pred, labels=list(range(3)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = None
    covid_idx = CLASS_NAMES.index("COVID-19")
    covid_total = int(cm[covid_idx].sum())
    covid_fn = int(cm[covid_idx].sum() - cm[covid_idx, covid_idx])
    metrics = {
        "test_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_roc_auc_ovr": macro_auc,
        "classification_report": report,
        "confusion_matrix_labels_order": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "covid19_false_negative_count": covid_fn,
        "covid19_actual_positive_count": covid_total,
        "covid19_false_negative_rate": (covid_fn / covid_total) if covid_total else None,
        "split_summary": split_info,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\\nSaved results/baseline_metrics.json")
=======
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataset import make_loaders, CLASS_NAMES
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_resnet18.pt")
    args = parser.parse_args()

    _, _, test_loader = make_loaders("data/xray_raw")
    model = build_model(3).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.softmax(model(images.to(DEVICE)), dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(probs.argmax(1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(confusion_matrix(y_true, y_pred))
    auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    covid_fn = cm[2, :].sum() - cm[2, 2]
    covid_total = cm[2, :].sum()
    print(f"COVID-19 false-negative rate: {covid_fn/covid_total:.4f}")
>>>>>>> origin/main

if __name__ == "__main__":
    main()
