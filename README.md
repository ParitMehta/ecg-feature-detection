# ECG Diagnostic Classification Using Time-Frequency Representations

## Problem Statement

Advancements in the field of health-tech have led to ECG screening powered by ML, that can catch cardiac conditions that would otherwise go undiagnosed until an emergency event, making large-scale preventive cardiology increasingly feasible. The best current classification models work
directly on the raw voltage signal. But an ECG also carries information in how
its frequency content changes over the course of a heartbeat, like the sharpness
of a QRS peak or the energy distribution across frequency bands, and standard
models have no reason to look for that structure unless it is given to them. One
way to surface it is to convert the signal into a spectrogram, a 2D image of
time versus frequency, and classify that image with a convolutional neural
network, the same architecture used in medical imaging. Another is to extract
known spectral features directly from the signal. It is unknown whether either
approach improves over the raw-signal baseline, whether combining them adds
anything, and whether the answer changes depending on the cardiac condition or
the quality of the recording. This project tests that on the PTB-XL dataset, and
separately asks how much accuracy survives when the models are compressed for
deployment in settings where patient data cannot leave the premises.


## Data

### Source and Access

PTB-XL v1.0.3, hosted on PhysioNet:
https://physionet.org/content/ptb-xl/1.0.3/

Open access, no application required. To download:

```bash
bash scripts/download_data.sh
```

This fetches the dataset into `data/ptbxl/`, which is gitignored.

### Structure

- 21,799 twelve-lead ECG recordings from 18,869 patients
- Each recording is 10 seconds long, sampled at 500 Hz
- Raw signals stored in WaveForm DataBase (WFDB) format
- Metadata in `ptbxl_database.csv`, one row per recording

### Labels

Each recording is annotated with one or more SCP-ECG diagnostic codes stored in
the `scp_codes` column as a dictionary mapping code to confidence (0-100). These
aggregate into five diagnostic superclasses:

| Superclass | Meaning |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

A single recording can carry multiple labels.

### Train/Validation/Test Splits

The dataset provides a `strat_fold` column (values 1-10) assigning each
recording to a fold. Splits are stratified by label distribution and grouped
by patient, meaning no patient appears in more than one fold.

- Folds 1-8: training
- Fold 9: validation
- Fold 10: test

These splits are fixed and must not be modified. Random splitting would cause
data leakage since some patients have multiple recordings.

## Reproducibility

### Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

### Setup

```bash
git clone https://github.com/ParitMehta/ecg-feature-detection.git
cd ecg-feature-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

### Project Structure

├── artifacts/ # saved models, experiment outputs
├── configs/ # experiment configuration files
├── data/ # raw and processed data (gitignored)
├── docs/ # additional documentation
├── notebooks/ # exploratory analysis
├── src/ # source code
├── tests/ # unit tests
├── .github/ # CI workflows
├── .gitignore
├── README.md
└── requirements.txt

## Methods
  - Baseline: Raw Signal Classification
  - Spectrogram Representation
  - Spectral Feature Extraction
  - Fusion Architecture
  - Model Compression

## Evaluation
  - Metrics
  - Per-Condition Breakdown

## Results

## Successes and failures

## Reproducibility
  - Requirements
  - Setup
  - Running Experiments

## Project Structure
