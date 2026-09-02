import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from dataset import make_loaders
from model import build_model

DATA_DIR = "data/xray_raw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    Path("models").mkdir(exist_ok=True)
    train_loader, val_loader, _ = make_loaders(DATA_DIR)
    model = build_model(3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    best_auc = -1
    for epoch in range(10):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.softmax(model(images.to(DEVICE)), dim=1)
                y_true.extend(labels.numpy())
                y_score.extend(probs.cpu().numpy())

        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        print(f"Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}, val_macro_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/best_resnet18.pt")

if __name__ == "__main__":
    main()
