# MedScan Diagnostics — Week 3 Checkpoint 2

**Project:** Three-class chest X-ray classification (NORMAL, PNEUMONIA, COVID-19) using ResNet-18.

This package implements the Week 3 ingestion/preprocessing pipeline, reproducible data splitting, baseline training, evaluation, and a baseline report.

## Important dataset note
The supplied Checkpoint 1 ZIP did not contain the 3,000 chest X-ray images described in the project brief. It contained a healthcare symptom CSV and healthcare test-case text file. Those files are preserved as supporting materials but are not suitable for training this image classifier. Therefore, no training/evaluation numbers are fabricated or claimed. The baseline report records this dependency transparently.

## 1. Expected image folder structure
Place the approved dataset in `data/xray_raw/`:

```text
data/xray_raw/
├── NORMAL/
├── PNEUMONIA/
└── COVID-19/
```

Each folder should contain the corresponding labelled X-ray images. Confirm that use and storage of the images is permitted. Do not commit private/identifiable medical images to a public repository.

## 2. Setup
Use Python 3.10 or 3.11 in a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Validate data and create split manifests

```bash
python src/prepare_data.py --data-dir data/xray_raw --seed 42
```

This checks that each class contains readable images, reports class counts and writes train/validation/test manifests in `results/`. The split is stratified at image level. If patient IDs are available, perform patient-level splitting instead to avoid leakage.

## 4. Train baseline

```bash
python src/train.py --data-dir data/xray_raw --epochs 10 --batch-size 32 --seed 42
```

The script uses ResNet-18, cross-entropy loss and AdamW. It saves the best validation-AUC checkpoint to `models/best_resnet18.pt` and training history to `results/training_history.json`.

If pretrained weights cannot be downloaded in your environment, use `--no-pretrained` (this changes the baseline setup and should be documented).

## 5. Evaluate on held-out test set

```bash
python src/evaluate.py --data-dir data/xray_raw --checkpoint models/best_resnet18.pt --seed 42
```

Writes `results/baseline_metrics.json` with accuracy, class-wise metrics, macro ROC-AUC, confusion matrix and COVID-19 false-negative rate.

## 6. Week 3 artifacts
- `src/dataset.py` — ingestion, image verification, transforms, stratified split and loaders.
- `src/prepare_data.py` — dataset validation and split manifest generation.
- `src/model.py` — ResNet-18 architecture.
- `src/train.py` — baseline training and checkpointing.
- `src/evaluate.py` — held-out evaluation.
- `BASELINE_REPORT.md` — benchmark methodology and status.
- `CHECKPOINT_2_SUBMISSION.md` — milestone summary.

## Acceptance targets (not measured results)
- Macro ROC-AUC ≥ 0.87.
- COVID-19 false-negative rate < 15%.

## Safety
Educational/research prototype only. Not for diagnosis, patient triage or treatment decisions. A model's output or Grad-CAM heatmap is not a clinical finding.
