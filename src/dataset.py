from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "COVID-19"]

def collect_samples(root):
    root = Path(root)
    samples = []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        for path in (root / class_name).glob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((str(path), label_idx))
    return samples

class XrayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms

def make_loaders(data_dir, batch_size=32, seed=42):
    samples = collect_samples(data_dir)
    labels = [y for _, y in samples]
    train, temp = train_test_split(samples, test_size=0.30,
                                   stratify=labels, random_state=seed)
    temp_labels = [y for _, y in temp]
    val, test = train_test_split(temp, test_size=0.50,
                                 stratify=temp_labels, random_state=seed)

    train_tfms, eval_tfms = get_transforms()
    return (
        DataLoader(XrayDataset(train, train_tfms), batch_size=batch_size, shuffle=True),
        DataLoader(XrayDataset(val, eval_tfms), batch_size=batch_size, shuffle=False),
        DataLoader(XrayDataset(test, eval_tfms), batch_size=batch_size, shuffle=False),
    )
