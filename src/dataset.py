from pathlib import Path
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "COVID-19"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def collect_samples(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image dataset folder not found: {root.resolve()}")
    samples, counts = [], {}
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        files = sorted(p for p in class_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if class_dir.exists() else []
        counts[class_name] = len(files)
        samples.extend((str(p), label_idx) for p in files)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            "No images found for class(es): " + ", ".join(missing) +
            f". Expected folders under {root.resolve()}. Counts: {counts}"
        )
    # Verify that files can be decoded before training.
    valid, bad = [], []
    for path, label in samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append((path, label))
        except (OSError, UnidentifiedImageError):
            bad.append(path)
    if bad:
        bad_file = Path("results") / "invalid_images.txt"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("\\n".join(bad), encoding="utf-8")
        print(f"Warning: skipped {len(bad)} unreadable image(s); see {bad_file}")
    counts_valid = {name: sum(y == i for _, y in valid) for i, name in enumerate(CLASS_NAMES)}
    if any(v < 3 for v in counts_valid.values()):
        raise ValueError(f"At least 3 readable images per class are required for stratified splitting. Counts: {counts_valid}")
    return valid

class XrayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, label

def get_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms

def split_samples(samples, seed=42):
    labels = [label for _, label in samples]
    train, temp = train_test_split(samples, test_size=0.30, stratify=labels, random_state=seed)
    temp_labels = [label for _, label in temp]
    val, test = train_test_split(temp, test_size=0.50, stratify=temp_labels, random_state=seed)
    return train, val, test

def make_loaders(data_dir="data/xray_raw", batch_size=32, seed=42, num_workers=0):
    samples = collect_samples(data_dir)
    train, val, test = split_samples(samples, seed)
    train_tfms, eval_tfms = get_transforms()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    loaders = (
        DataLoader(XrayDataset(train, train_tfms), shuffle=True, **common),
        DataLoader(XrayDataset(val, eval_tfms), shuffle=False, **common),
        DataLoader(XrayDataset(test, eval_tfms), shuffle=False, **common),
    )
    split_summary = {
        "total": len(samples), "train": len(train), "validation": len(val), "test": len(test),
        "class_counts": {name: sum(y == i for _, y in samples) for i, name in enumerate(CLASS_NAMES)}
    }
    return (*loaders, split_summary)
