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

if __name__ == "__main__":
    main()
