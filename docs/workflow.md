# InfraCrackVision — Phased Project Workflow

## Project objective

InfraCrackVision will accept a concrete, pavement, bridge-deck, or wall image and classify it into four classes:

1. No crack
2. Minor crack
3. Moderate crack
4. Severe crack

The primary system will be a CNN. Where pixel-level masks are available, the CNN will also produce a crack segmentation mask. The minor/moderate/severe labels are initially **crack-coverage proxy labels**, not expert structural-risk labels.

---

## Phase 0 — Freeze the scope and protocol

Before downloading or training anything:

- Confirm the four output classes.
- Confirm that the CNN is the primary model.
- Define ANN and QNN as secondary experiments.
- Define the image-only inference goal.
- Define the train/validation/test policy.
- Define the evaluation metrics.
- Record dataset licenses and sources.

**Deliverable:** approved project scope and experiment protocol.

---

## Phase 1 — Create the environment and project structure

Tasks:

- Create the `InfraCrackVision` project folder.
- Create and activate `.venv`.
- Install `requirements.txt`.
- Verify PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, and PennyLane.
- Verify that all source modules compile.
- Keep raw data outside Git and outside the starter ZIP.

**Deliverable:** reproducible project environment.

---

## Phase 2 — Acquire and inspect datasets

Initial datasets:

- DeepCrack
- SDNET2018

The larger Crack Segmentation Dataset is deferred for now because its archive is approximately 2 GB. It may be added later as an expansion, not as a prerequisite for the first working system.

For every dataset, record:

- Download source and license
- Number of images
- Number of masks
- Image dimensions and channels
- Class names, if provided
- Whether masks exist
- Original scene or source-image identity
- Duplicate or near-duplicate risk

Do not combine datasets yet.

**Deliverable:** dataset inspection report and source-specific file counts.

---

## Phase 3 — Standardize dataset storage

Keep source datasets separate:

```text
datasets/
├── deepcrack/
│   ├── images/
│   └── masks/
├── sdnet2018/
│   ├── cracked/
│   └── noncracked/
└── crack_segmentation/
    ├── images/
    └── masks/
```

Do not overwrite original files. Create processed copies only when required.

**Deliverable:** clean, traceable dataset directories.

---

## Phase 4 — Build the unified metadata table

Create one metadata file containing:

```text
filename
source_dataset
image_path
mask_path
has_crack
mask_available
crack_pixel_ratio
severity_class
source_scene_id
split
```

Rules:

- Explicit non-crack labels become `no-crack`.
- Images with masks can receive crack-coverage labels.
- Cracked images without masks must not be assigned minor/moderate/severe automatically.
- Preserve the source dataset and original scene identity.

**Deliverable:** `results/metadata.csv`.

---

## Phase 5 — Create the four-class labels

For every image with a mask:

\[
\text{crack ratio} =
\frac{\text{white crack-mask pixels}}
{\text{total mask pixels}}
\]

Label policy:

```text
No crack:  explicit non-crack label or zero mask ratio
Minor:     lower third of cracked training ratios
Moderate:  middle third of cracked training ratios
Severe:    upper third of cracked training ratios
```

The minor/moderate/severe thresholds must be calculated from training data only and reused unchanged for validation and test data.

**Deliverable:**

```text
results/labels.csv
results/label_config.json
```

---

## Phase 6 — Analyze the data before training

Generate:

- Four-class distribution
- Dataset-source distribution
- Image-size distribution
- Crack-ratio histogram
- Sample image grid
- Sample mask grid
- Missing-mask report
- Duplicate/near-duplicate report

Decision point:

- If one class is severely underrepresented, decide whether to use class weights, oversampling, or additional data.
- Do not change the test distribution to make the result look better.

**Deliverable:** data analysis report and decision log.

---

## Phase 7 — Create source-safe train/validation/test splits

Preferred split:

```text
Training set:   70–80%
Validation set: 10–15%
Test set:       15–20%
```

Split by original scene, source image, or physical location where possible. Do not place crops from the same original image in different splits.

The test set must remain untouched until final evaluation.

**Deliverable:** fixed split metadata with a reproducible random seed.

---

## Phase 8 — Build the primary four-class CNN classifier

Input:

```text
RGB image
```

Output:

```text
No crack / Minor / Moderate / Severe
```

Use:

- CNN backbone
- Resize and normalization
- Data augmentation applied consistently
- Cross-entropy or class-weighted cross-entropy
- Best-checkpoint saving based on validation macro F1

**Deliverable:** working classifier and smoke-test checkpoint.

---

## Phase 9 — Add the CNN segmentation branch

Where masks are available, add a segmentation output:

```text
CNN features → predicted binary crack mask
```

Use:

- Weighted BCE or focal loss
- Dice loss if appropriate
- Combined classification and segmentation loss

The final CNN can produce:

```text
Four-class severity prediction
+
Predicted crack mask
```

**Deliverable:** segmentation-capable CNN and example overlays.

---

## Phase 10 — Run smoke tests

Before full training:

- Use 2–3 epochs.
- Use a small, representative subset.
- Confirm all classes are represented if possible.
- Check that losses are finite.
- Check that checkpoints save and reload.
- Check that plots and metrics are generated.
- Test one new image through inference.

**Deliverable:** smoke-test report. Full training starts only after this phase passes.

---

## Phase 11 — Full CNN training

Train using the fixed training split and monitor only the validation split.

Save:

```text
checkpoints/cnn_best.pt
results/history_cnn.json
results/curves/cnn_*.png
```

Do not use the test set during training or model selection.

**Deliverable:** trained primary CNN checkpoint.

---

## Phase 12 — Final CNN evaluation

Evaluate once on the untouched test set.

Classification metrics:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Macro F1
- Per-class F1
- Confusion matrix

Segmentation metrics:

- IoU
- Dice score
- Pixel precision
- Pixel recall

**Deliverable:** final CNN evaluation report.

---

## Phase 13 — Add the ANN baseline

The ANN will use only raw-image features:

- Intensity statistics
- Edge density
- Gradient statistics
- Texture statistics

It must not use:

- Crack-pixel ratio
- Mask pixel counts
- Any mask-derived input feature

**Deliverable:** leakage-safe ANN results for the same fixed split.

---

## Phase 14 — Add the QNN experiment

The QNN will use reduced raw-image features with:

- Four to eight qubits maximum
- Angle embedding
- Variational entangling layers
- CPU simulation

Use the same labels and split as the CNN and ANN. Save the feature scaler and circuit weights.

**Deliverable:** reproducible QNN experiment and limitations report.

---

## Phase 15 — Build the inference pipeline

Command-line usage:

```bash
python inference/predict.py --image path/to/new_image.jpg
```

The system should:

1. Load a new RGB image.
2. Apply training-time preprocessing.
3. Predict one of the four classes.
4. Report confidence.
5. Produce a crack mask for the CNN.
6. Save an overlay image.
7. Optionally report predicted crack coverage.

**Deliverable:** new-image inference workflow.

---

## Phase 16 — Add the optional web interface

Only after command-line inference is reliable:

- Add an image-upload interface.
- Display the predicted class.
- Display confidence.
- Display the mask and overlay.
- Clearly label the result as visual screening, not structural certification.

**Deliverable:** optional local web demonstration.

---

## Phase 17 — Final documentation and presentation outputs

Prepare:

- Architecture diagram
- Workflow diagram
- Dataset table
- Labeling explanation
- Training curves
- Confusion matrices
- Segmentation examples
- Per-class metric table
- Limitations
- Future work
- Reproducibility instructions

**Deliverable:** final report, presentation figures, and reproducible project archive.
