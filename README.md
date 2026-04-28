# ECG Diagnostic Classification Using Time-Frequency Representations

## Problem Statement

Advancements in the field of health-tech have led to ECG screening powered by ML, that can catch cardiac conditions that would otherwise go undiagnosed until an emergency event, making large-scale preventive cardiology increasingly feasible. The best current classification models work
directly on the raw voltage signal. But an ECG also carries information in how
its frequency content changes over the course of a heartbeat, like the sharpness
of a QRS peak or the energy distribution across frequency bands, and standard
models have no reason to look for that structure unless it is given to them. One
way to surface it is to convert the signal into a spectrogram, a 2D image of
time versus frequency, and classify that image with a convolutional neural
network, the same architecture used in medical imaging. 

This project classifies 12-lead ECG recordings from the PTB-XL dataset into five diagnostic superclasses (NORM, MI, STTC, CD, HYP) and audits performance variation across patient sex and age groups. The pipeline runs as a numbered sequence of scripts (EDA → preprocessing → training → inspection → fairness → demo) and ships with a Streamlit app for single-recording inference.
Most published PTB-XL work reports a single aggregate AUROC; this project additionally benchmarks a 1D-CNN baseline, evaluates per-class performance at val-tuned decision thresholds, and measures subgroup performance across sex and age via a dedicated fairness audit.

Future aims: Another way is to extract known spectral features directly from the signal. It is unknown whether either approach improves over the raw-signal baseline, whether combining them adds
anything, and whether the answer changes depending on the cardiac condition or
the quality of the recording. This project will test that on the PTB-XL dataset, and
separately asks how much accuracy survives when the models are compressed for
deployment in settings where patient data cannot leave the premises.




---

## What's Different About This Project

- **Subgroup fairness analysis.** Every metric is recomputed separately for male vs. female patients and for four age bands (<40, 40–60, 60–75, 75+) using val-tuned thresholds.
- **Threshold-aware evaluation.** Per-class decision thresholds are tuned on the validation fold via an F1-argmax sweep over 0.05–0.95 (step 0.01). F1, precision, recall, and confusion matrices are all reported at those thresholds rather than only threshold-free AUROC.

---

## Data

### Source

PTB-XL v1.0.3 from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/

The loader reads `ptbxl_database.csv` and `scp_statements.csv`, retains only SCP-ECG codes with `diagnostic == 1`, and maps each recording to its list of diagnostic superclasses via the `diagnostic_class` column.

### Download

```bash
bash scripts/download_data.sh
```

The loader expects 100 Hz WFDB records under `data/physionet.org/files/ptb-xl/1.0.3/records100/` (and `records500/` for 500 Hz).

### Labels (five superclasses)

`NORM`, `MI`, `STTC`, `CD`, `HYP`, produced by `build_diagnosis_matrix` via `sklearn.preprocessing.MultiLabelBinarizer` with that fixed class order, so `class_names.txt` always matches the column order of `y_*.npy`. A single recording can carry multiple labels — this is a multi-label task.

### Train / Validation / Test Splits

PTB-XL ships with a `strat_fold` column (1–10). `split_by_fold` uses folds 1–8 for training, fold 9 for validation, and fold 10 for the held-out test. Both fold numbers are exposed as keyword arguments (`val_fold=9`, `test_fold=10`) on the pipeline call.

---

## Project Structure

```text
ecg-feature-detection/
├── src/
│   ├── paths.py        # DATASET, PROCESSED, REPORTS, MLRUNS + per-stage dirs
│   ├── loader.py       # load_metadata / add_diagnostic_labels / load_raw_signals
│   ├── preprocess.py   # keep_patients_with_recordings / clean_ecg_signals /
│   │                   # build_diagnosis_matrix / split_by_fold
│   └── model.py        # ECGNet (1D-CNN)
├── scripts/
│   ├── 01_eda.py
│   ├── 02_preprocess.py
│   ├── 03_train.py
│   ├── 04_inspect.py
│   ├── 05_fairness.py
│   ├── 06_app.py
│   └── download_data.sh
├── configs/
├── docs/
├── notebooks/
├── tests/
├── reports/
│   ├── eda/            # per-stage PNGs + CSVs from 01_eda.py
│   ├── preprocess/     # pipeline_dag.png from 02_preprocess.py
│   ├── train/          # best_model.pt, thresholds.json, report.txt
│   ├── inspect/        # metrics_per_class.csv, confusion matrices, etc.
│   └── fairness/       # subgroup_metrics.csv, metrics_by_sex/age.png
├── artifacts/
├── mlruns/             # MLflow store (file:///…/mlruns)
├── requirements.txt
└── README.md
```

All per-stage output directories (`PROCESSED`, `EDA`, `PREPROCESS`, `TRAIN`, `INSPECT`, `FAIRNESS`) are created automatically at import time by `src/paths.py`.

---

## Pipeline Visualization

The pipeline DAG is generated by the [`pipefunc`](https://github.com/pipefunc/pipefunc) library at runtime rather than maintained as a static diagram.

- `scripts/01_eda.py` calls `pipeline.visualize()` to render the EDA DAG inline.
- `scripts/02_preprocess.py` calls `pipeline.visualize_graphviz(filename=REPORT_FOLDER / "pipeline_dag.png")` and falls back to `pipeline.visualize()` if the system Graphviz binary is absent.

The canonical committed diagram is `reports/preprocess/pipeline_dag.png`, regenerated whenever `python scripts/02_preprocess.py` is executed. Dependencies: system Graphviz (`apt-get install graphviz` / `brew install graphviz`) plus `pip install "pipefunc[all]"`.

![Pipeline DAG](reports/preprocess/pipeline_dag.png)

---
## Pipeline — File-by-File

The pipeline is split into six numbered scripts that must be run in order. Each script
is self-contained: it reads from the outputs of the previous step and writes its own
outputs to a dedicated subfolder under `reports/`.

---

### `scripts/01_eda.py` — Exploratory Data Analysis

**Purpose:** Understand the dataset before touching it — class balance, label overlap,
signal quality, and how recordings are distributed across the ten folds.

Run this first to confirm the data downloaded correctly and to get a visual sense of
what the five diagnostic classes look like as raw ECG waveforms.

```bash
python scripts/01_eda.py
```

**What it does:**

- Counts recordings per diagnostic class and flags any recordings with no diagnostic label
- Plots lead-I waveforms for the first three patients
- Shows how many labels each recording carries (some recordings have more than one diagnosis)
- Plots the class distribution across all ten folds to verify stratification
- Exports one representative ECG per superclass and a full 12-lead plot for the first patient
- Runs a sanity check on 100 random recordings: checks for NaNs, voltage range, and
  whether any lead is near-silent

**Outputs in `reports/eda/`:**

| File | Description |
|---|---|
| `class_counts.csv` / `01_class_counts.png` | Recording count per superclass |
| `02_lead_I_first_three.png` | Lead-I waveforms for patients 1–3 |
| `labels_per_record.csv` | Label-cardinality distribution |
| `per_fold_distribution.csv` / `03_per_fold_distribution.png` | Class balance across folds |
| `04_one_record_per_superclass.png` | One representative ECG per class |
| `05_all_12_leads.png` | All 12 leads of the first patient |
| `eda_summary.txt` | Plain-text summary of all findings |

---

### `scripts/02_preprocess.py` — Preprocessing

**Purpose:** Convert the raw WFDB recordings into clean NumPy arrays that the model
can train on, and produce the train/validation/test split files.

```bash
python scripts/02_preprocess.py
```

**What it does:**

1. Loads metadata and filters out recordings with no diagnostic label
2. Reads the raw 100 Hz ECG signals from disk
3. Cleans each signal with a 3rd-order Butterworth bandpass filter (0.5–40 Hz) to
   remove baseline wander and high-frequency noise, then z-scores each lead independently
4. Converts the SCP-ECG diagnostic codes into a binary label matrix
   (`NORM`, `MI`, `STTC`, `CD`, `HYP`), one column per class
5. Splits recordings into train (folds 1–8), validation (fold 9), and test (fold 10)
6. Saves each split as `.npy` files ready for training

Also renders the full pipeline as a DAG diagram at `reports/preprocess/pipeline_dag.png`.

**Writes to `data/processed/`:**

`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, `class_names.txt`

---

### `scripts/03_train.py` — Training

**Purpose:** Train the 1D-CNN (`ECGNet`) on the preprocessed splits, select the best
checkpoint, tune per-class decision thresholds on the validation set, and evaluate on
the held-out test fold.

```bash
python scripts/03_train.py
```

**What it does:**

1. Trains `ECGNet` for 10 epochs using binary cross-entropy loss with per-class
   positive-weight correction to handle class imbalance
2. Saves the checkpoint with the highest validation macro-AUROC as `best_model.pt`
3. Tunes a separate decision threshold for each class on the validation set by sweeping
   over thresholds from 0.05 to 0.95 and selecting the one that maximises F1 — this step
   matters because a fixed 0.5 threshold performs poorly on imbalanced classes like HYP
4. Evaluates the best checkpoint on the test fold using the tuned thresholds
5. Logs all hyperparameters, metrics, and artifacts to MLflow

**Outputs in `reports/train/`:**

| File | Description |
|---|---|
| `best_model.pt` | Saved model weights |
| `thresholds.json` | Val-tuned decision threshold per class |
| `report.txt` | Best val AUROC, test AUROC, and per-class AUROC |

---

### `scripts/04_inspect.py` — Model Inspection

**Purpose:** Produce a detailed breakdown of model performance on the test set — not
just a single number, but per-class metrics, confusion matrices, and concrete examples
of correct and incorrect predictions.

```bash
python scripts/04_inspect.py
```

**What it does:**

1. Loads the trained model and the val-tuned thresholds from `reports/train/`
2. Runs inference on the full test fold
3. Reports AUROC, F1, precision, recall, and support for each class individually,
   plus macro averages
4. Renders a 2×2 confusion matrix for each class at its tuned threshold
5. Exports a co-occurrence matrix showing which pairs of classes are predicted together
6. Saves a gallery of six example patients — three where predictions exactly match the
   ground-truth labels, and three where they do not — plotted on lead II

**Outputs in `reports/inspect/`:**

| File | Description |
|---|---|
| `metrics_per_class.csv` | AUROC/F1/precision/recall/support per class |
| `per_class_auroc.png` | Bar chart of per-class AUROC |
| `confusion_matrices.png` | 2×2 confusion matrix per class |
| `pred_cooccurrence.csv` | Prediction co-occurrence counts |
| `predictions_head.csv` | First 20 rows of predicted vs. true labels |
| `patient_XX_idxYYYY.png` | Example correct / incorrect patient plots |

---

### `scripts/05_fairness.py` — Fairness Audit

**Purpose:** Check whether the model performs equally well across demographic groups.
A model with a strong overall AUROC can still systematically underperform for specific
patient subgroups — this script surfaces those gaps.

```bash
python scripts/05_fairness.py
```

**What it does:**

1. Reconstructs the test-fold patient metadata (sex, age) and aligns it with `X_test`
2. Bins patients into six subgroups: male, female, and four age bands
   (<40, 40–60, 60–75, 75+)
3. Runs inference at the val-tuned thresholds and reports macro and per-class
   AUROC/F1/precision/recall for each subgroup separately

**Outputs in `reports/fairness/`:**

| File | Description |
|---|---|
| `subgroup_metrics.csv` | Full macro + per-class metrics for every subgroup |
| `metrics_by_sex.png` | Metric comparison: male vs. female |
| `metrics_by_age.png` | Metric comparison across the four age bands |

---

### `scripts/06_app.py` — Streamlit Demo

**Purpose:** Provide an interactive interface for exploring model predictions on
individual ECG recordings — useful for understanding model behaviour without writing code.

```bash
streamlit run scripts/06_app.py
```

**Prerequisites:** `reports/train/best_model.pt` and the processed test-set files
(`X_test.npy`, `y_test.npy`, `class_names.txt`) must exist. Run `02_preprocess.py`
and `03_train.py` first.

**Three input modes (sidebar radio):**

| Mode | What to provide |
|---|---|
| Browse test set | Select any row from the preprocessed test split by index |
| Upload WFDB record | Upload a paired `.hea` + `.dat` file from the PTB-XL dataset |
| Upload .npy array | Upload a raw `(1000, 12)` float array |

The main pane shows predicted vs. true labels, a per-class probability bar chart, and
all 12 leads in a 6×2 grid. A sidebar threshold slider (default 0.5) overrides the
val-tuned thresholds for live exploration. A fairness panel at the bottom displays the
sex and age breakdown plots from `reports/fairness/` if they exist.

### `src/` — Shared Library

These files are imported by the pipeline scripts and do not need to be run directly.
They exist so that common logic (file paths, data loading, signal cleaning, the model
definition) is written once and shared across all six scripts rather than duplicated.

- **`paths.py`** — Defines where everything lives on disk: the raw dataset, the
  processed arrays, and the output folders for each pipeline stage. All output
  folders are created automatically the first time any script is run, so nothing
  needs to be created manually.

- **`loader.py`** — Handles reading the PTB-XL files from disk: the patient
  metadata spreadsheet, the diagnostic code reference table, and the raw ECG signal
  files. The scripts call these functions rather than dealing with file formats directly.

- **`preprocess.py`** — Contains the signal cleaning and data preparation logic:
  filtering noise from the ECG signals, normalising each lead, converting diagnosis
  codes into a label matrix, and splitting recordings into train/validation/test sets.

- **`model.py`** — Defines the neural network (`ECGNet`). See the architecture
  section below.

#### Model architecture

`ECGNet` is a 1D convolutional neural network. It takes a 10-second, 12-lead ECG as
input and outputs a probability for each of the five diagnostic classes. The key design
decisions are:

- **1D convolutions** operate along the time axis, detecting local waveform patterns
  (e.g. a QRS complex) across all 12 leads at once.
- **Three convolutional blocks** progressively increase the number of filters (32 → 64
  → 128) while halving the time resolution at each step via max-pooling, allowing later
  layers to detect longer-range patterns.
- **Global average pooling** collapses the entire time axis into a single 128-dimensional
  vector before the final classification layer, making the model robust to slight
  differences in recording length.
- **Five independent outputs** — one logit per class — so a recording can be positive
  for any combination of classes simultaneously.

Input → (batch, 12 leads, 1000 time steps @ 100 Hz)
│
▼
Block 1: Conv1d → BatchNorm → ReLU → MaxPool → (batch, 32, 500)
Block 2: Conv1d → BatchNorm → ReLU → MaxPool → (batch, 64, 250)
Block 3: Conv1d → BatchNorm → ReLU → MaxPool → (batch, 128, 125)
│
▼
GlobalAvgPool → (batch, 128)
Linear → (batch, 5 logits)
│
▼
Output → sigmoid → probability per class

text

---

## MLflow Experiment Tracking

MLflow records every training run automatically — hyperparameters, loss curves, and
final metrics — so results are reproducible and comparable across runs.

After training, browse the run history in a local web UI:

```bash
mlflow ui --backend-store-uri ./mlruns
# then open http://localhost:5000
```

Everything logged per run:

- All training hyperparameters (learning rate, batch size, etc.)
- Loss and AUROC at every epoch, for both the training and validation sets
- Final test AUROC overall and per class
- The saved model weights and the val-tuned thresholds file

---

## Streamlit Demo

The demo app lets you run the trained model on any ECG recording through a browser
interface — no code required.

### Before launching

Two pipeline steps must have been run first:

```bash
python scripts/02_preprocess.py   # produces the test-set arrays
python scripts/03_train.py        # produces the trained model
```

### Launch

```bash
streamlit run scripts/06_app.py
# then open http://localhost:8501
```

### How to use it

Choose an input source from the sidebar:

| Option | Use when… |
|---|---|
| **Browse test set** | Exploring model behaviour on recordings it has already been evaluated on. Ground-truth labels are shown for comparison. |
| **Upload WFDB record** | Testing on a new PTB-XL recording. Upload the `.hea` and `.dat` files together. |
| **Upload .npy array** | Testing on a custom recording exported as a NumPy array of shape `(1000, 12)`. |

The sidebar also has a **threshold slider** (default 0.5). Lowering it makes the model
more sensitive — it will flag more recordings as positive for a given class but will
also produce more false positives. The val-tuned thresholds in `thresholds.json` are
the values that maximised F1 on the validation set and are the recommended defaults for
evaluation.
## Evaluation

- **Primary metric:** macro-averaged AUROC across the five superclasses on the held-out test fold (fold 10).
- **Per-class metrics at tuned thresholds:** AUROC, F1, precision, recall, support, plus a `macro` row — in `reports/inspect/metrics_per_class.csv`.
- **Confusion matrices:** one 2×2 per class at val-tuned thresholds — `reports/inspect/confusion_matrices.png`.
- **Subgroup metrics:** macro + per-class AUROC/F1/precision/recall per subgroup — `reports/fairness/subgroup_metrics.csv`.

---

## Results

### Overall (test set, fold 10)

- Best validation macro-AUROC: **0.9062**
- Test macro-AUROC: **0.9043**

Per-class test metrics at val-tuned thresholds:

| Class     | Threshold | AUROC      | F1     | Precision | Recall | Support |
|-----------|-----------|------------|--------|-----------|--------|---------|
| NORM      | 0.31      | 0.9353     | 0.8432 | 0.7695    | 0.9325 | 963     |
| MI        | 0.81      | 0.9111     | 0.7121 | 0.6943    | 0.7309 | 550     |
| STTC      | 0.68      | 0.9279     | 0.7552 | 0.7133    | 0.8023 | 521     |
| CD        | 0.62      | 0.9146     | 0.7320 | 0.6991    | 0.7681 | 496     |
| HYP       | 0.67      | 0.8325     | 0.4648 | 0.4069    | 0.5420 | 262     |
| **macro** | —         | **0.9043** | **0.7015** | **0.6566** | **0.7552** | 2792 |

![Per-class AUROC](reports/inspect/per_class_auroc.png)
![Confusion matrices](reports/inspect/confusion_matrices.png)

### Subgroup Performance

| Group      | n    | Macro AUROC | Macro F1 | Macro Precision | Macro Recall |
|------------|------|-------------|----------|-----------------|--------------|
| male       | 1132 | 0.913       | 0.717    | 0.677           | 0.762        |
| female     | 1066 | 0.895       | 0.686    | 0.638           | 0.748        |
| age <40    | 310  | 0.904       | 0.553    | 0.590           | 0.531        |
| age 40–60  | 657  | 0.898       | 0.677    | 0.655           | 0.701        |
| age 60–75  | 723  | 0.895       | 0.697    | 0.652           | 0.752        |
| age 75+    | 474  | 0.867       | 0.672    | 0.596           | 0.784        |

![Metrics by sex](reports/fairness/metrics_by_sex.png)
![Metrics by age](reports/fairness/metrics_by_age.png)

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/ParitMehta/ecg-feature-detection.git
cd ecg-feature-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For pipeline DAG export in `02_preprocess.py`, the system Graphviz binary is also required (`apt-get install graphviz` / `brew install graphviz`) along with `pip install "pipefunc[all]"`.

### 2. Download the data

```bash
bash scripts/download_data.sh
```

Places PTB-XL v1.0.3 under the path expected by `src/paths.py` (`DATASET = data/physionet.org/files/ptb-xl/1.0.3`).

### 3. Run the pipeline in order

```bash
python scripts/01_eda.py
python scripts/02_preprocess.py
python scripts/03_train.py
python scripts/04_inspect.py
python scripts/05_fairness.py
streamlit run scripts/06_app.py
```

Each step depends on outputs produced by the previous one.

---

## Limitations

- Research only — not a medical device.
- PTB-XL was collected at a single German institution; results may not generalise to other populations or recording hardware.
- Model outputs are raw sigmoid probabilities, not calibrated probabilities.
- Subgroup performance varies across sex and age groups; any deployment consideration must first address the gaps documented in `reports/fairness/subgroup_metrics.csv`.
