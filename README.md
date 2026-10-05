# InfraCrackVision

### Deep-learning-based visual crack detection and crack-coverage severity estimation for concrete and pavement structures

InfraCrackVision is an academic deep-learning project that processes RGB images of concrete, pavement, bridge-deck, and wall surfaces. The system is designed to identify visible cracks, localize them with a predicted binary mask, and estimate a four-class visual severity category:

```text
no-crack | minor | moderate | severe
```

The project is intended as a visual screening and maintenance-prioritization tool. It is not a substitute for a qualified structural engineer or a certified structural-safety assessment.

---

## Project objective

Given a new RGB image, InfraCrackVision aims to produce:

- A four-class crack-severity prediction.
- A confidence score and class probabilities.
- A predicted crack mask.
- Estimated visible crack coverage.
- An overlay image showing the predicted crack region.
- A result that can support inspection prioritization.

The current severity labels are **crack-coverage proxy labels**. The minor, moderate, and severe classes are derived from pixel-level mask coverage where masks are available. They are not expert-validated structural-risk categories.

---

## Research foundation

The CNN segmentation design is inspired by:

> Y. Liu et al., “DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation,” *Neurocomputing*, vol. 338, pp. 139–153, 2019.

The original DeepCrack task is binary pixel-level segmentation. InfraCrackVision extends the workflow toward four-class visible crack-coverage severity estimation and new-image inference.

---

## System workflow

```text
RGB surface image
        ↓
Image validation and preprocessing
        ↓
CNN feature extraction
        ↓
Four-class severity prediction
        ↓
Predicted crack segmentation mask
        ↓
Visible crack-coverage estimate
        ↓
Class, confidence, mask, overlay, and inspection suggestion
```

The implementation is developed in independent phases:

1. Dataset validation
2. Unified metadata and labeling
3. Source-safe train/validation/test splitting
4. CNN segmentation
5. Four-class CNN classification
6. Held-out evaluation
7. New-image inference
8. Optional ANN/QNN supporting experiments

The complete workflow is documented in [`docs/workflow.md`](docs/workflow.md).

---

## Repository structure

```text
InfraCrackVision/
├── data/                  # Dataset inspection, metadata, labels, and splits
├── models/                # CNN, segmentor, ANN, and QNN definitions
├── training/              # Segmentation and four-class CNN training
├── evaluation/            # Held-out test evaluation
├── inference/             # New-image inference
├── docs/                  # Workflow, labeling, architecture, and protocol
├── datasets/              # Local datasets; excluded from Git
├── checkpoints/           # Local model weights; excluded from Git
├── results/               # Generated outputs; excluded from Git
├── requirements.txt
└── README.md
```

---

## Datasets

### DeepCrack

DeepCrack provides RGB images and pixel-level binary masks. It is used for:

- Crack segmentation.
- Image-mask verification.
- Crack-pixel-ratio calculation.
- Minor/moderate/severe proxy-label generation.

Expected local layout:

```text
datasets/deepcrack/
├── images/
│   ├── train/
│   └── test/
└── masks/
    ├── train/
    └── test/
```

The verified dataset contains 300 training image-mask pairs and 237 testing image-mask pairs.

### SDNET2018

SDNET2018 is used to expand the visual diversity and no-crack class. Its cracked images do not generally include ground-truth masks. Therefore:

- Non-cracked images are assigned `no-crack`.
- Cracked images without masks are initially marked `severity_pending`.
- Pseudo-masks may be generated experimentally, but they are not treated as expert ground truth.

Expected local layout:

```text
datasets/sdnet2018/
├── cracked/
├── noncracked/
└── pseudo_masks/
```

The raw datasets are intentionally excluded from this repository because of their size and licensing requirements.

---

## Environment setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset and metadata workflow

Run these commands from the repository root after placing the datasets locally:

```powershell
python data\validate_dataset.py
python data\build_metadata.py
python data\prepare_labels.py
python data\filter_pseudo_labels.py
python data\build_four_class_metadata.py
python data\split_by_source.py
```

The scripts generate local files under `results/`, including:

```text
metadata.csv
labels.csv
label_config.json
sdnet_pseudo_labels_filtered.csv
four_class_metadata.csv
splits.csv
```

The test set is kept separate from training and validation. DeepCrack test images remain test-only.

---

## Training

### Train the segmentation model

Smoke test:

```powershell
python training\train_segmentor.py --epochs 2 --limit 32
```

Controlled training run:

```powershell
python training\train_segmentor.py --epochs 30
```

Checkpoint:

```text
checkpoints/segmentor.pt
```

### Train the four-class CNN

Smoke test:

```powershell
python training\train_four_class.py --epochs 2 --limit 64
```

Controlled prototype training:

```powershell
python training\train_four_class.py --epochs 20 --limit 4000
```

The controlled run uses approximately 1,000 samples per class and prevents the large pseudo-labeled classes from dominating the first experiment.

Checkpoint:

```text
checkpoints/four_class_cnn_best.pt
```

The best checkpoint is selected using validation macro F1, not test accuracy.

---

## Evaluation

Evaluate the saved four-class CNN on the untouched test split:

```powershell
python evaluation\evaluate_classifier.py
```

Generated output:

```text
results/four_class_test_metrics.json
```

The evaluation reports:

- Accuracy
- Balanced accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Macro F1-score
- Confusion matrix

Segmentation evaluation is a separate task and should report:

- Intersection over Union (IoU)
- Dice score
- Pixel precision
- Pixel recall

---

## New-image inference

After the classifier and segmentor checkpoints exist:

```powershell
python inference\predict.py --image "C:\path\to\new_image.jpg"
```

Example output files are saved under:

```text
results/inference/
├── image_prediction.json
├── image_mask.png
└── image_overlay.png
```

The JSON report contains:

- Predicted class.
- Confidence.
- Class probabilities.
- Predicted crack coverage.
- Mask path.
- Overlay path.

For poor-quality or low-confidence images, the appropriate outcome is manual engineering review rather than an unconditional safety decision.

---

## Data-leakage and evaluation policy

InfraCrackVision follows these rules:

- Crack-pixel ratio is never supplied as an input to the ANN classifier.
- Ground-truth masks are not supplied as classifier inputs.
- Label cutoffs are computed from training masks only.
- DeepCrack test images remain test-only.
- The test set is not used for checkpoint selection.
- Metrics are reported separately for classification and segmentation.
- Pseudo-labels are clearly distinguished from ground-truth labels.
- Dataset source and scene identity are preserved where possible.

---

## Limitations

The current system has several known limitations:

- Minor/moderate/severe are crack-coverage proxy classes.
- Pseudo-masks may contain segmentation errors.
- Dataset domain shift may occur between DeepCrack and SDNET2018.
- A single image cannot determine crack depth, structural load, reinforcement damage, or complete structural safety.
- External-camera generalization requires further validation.
- Expert civil-engineering labels would be required for certified severity assessment.

---

## Git and data policy

The following are intentionally excluded from Git:

```text
.venv/
datasets/
checkpoints/
results/
*.zip
*.pt
*.pth
__pycache__/
*.pyc
```

Each team member should clone the repository, create a local environment, and place the datasets locally according to the documented structure.

---

## Team

- Batch No. 16
- K Sravan Yadav — 24BCA7240
- S P Vaibhav — 24BCA7772
- K P S Ravi Shashank — 24BCE8086

Course: **CSE 4006 — Deep Learning**  
Semester: **Fall 2026–2027**
