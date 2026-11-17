import argparse, json
from pathlib import Path
from collections import Counter
from dataset import collect_samples, split_samples, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser(description="Validate image folders and create reproducible split manifests")
    parser.add_argument("--data-dir", default="data/xray_raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = collect_samples(args.data_dir)
    train, val, test = split_samples(samples, args.seed)
    out = Path("results"); out.mkdir(exist_ok=True)
    for name, rows in [("train", train), ("validation", val), ("test", test)]:
        payload = [{"image_path": p, "label": CLASS_NAMES[y]} for p, y in rows]
        (out / f"{name}_manifest.json").write_text(json.dumps(payload, indent=2))
        print(name, len(rows), dict(Counter(CLASS_NAMES[y] for _, y in rows)))
    print(f"Validated {len(samples)} readable images. Split manifests saved in {out}/")

if __name__ == "__main__":
    main()
