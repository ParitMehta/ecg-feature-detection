#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — build the train / validation / test splits.

Runs the full loader + preprocessing pipeline end-to-end and saves the
cleaned signals and 0/1 diagnosis labels as .npy files so the next
script (03_train.py) can load them instantly without re-running the
slow waveform decoding step.

Reads:
    {DATASET_FOLDER}/ptbxl_database.csv
    {DATASET_FOLDER}/scp_statements.csv
    {DATASET_FOLDER}/records100/*.hea + .dat       (21,799 ECGs)

Writes (to PROCESSED_FOLDER):
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    class_names.txt

And to REPORT_FOLDER:
    pipeline_dag.png          (rendered pipefunc diagram)
"""

#%% Imports and project paths
import sys
from pathlib import Path

sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
from pipefunc import Pipeline

# Loader stage  (definitions live in src/loader.py)
from src.loader import (
    load_metadata,
    add_diagnostic_labels,
    load_raw_signals,
)

# Preprocessing stage  (definitions live in src/preprocess.py)
# `keep_patients_with_recordings` was moved here so it is shared
# between 01_eda.py and this script (single source of truth).
from src.preprocess import (
    keep_patients_with_recordings,
    clean_ecg_signals,
    build_diagnosis_matrix,
    split_by_fold,
)

from src.paths import DATASET, PROCESSED, REPORTS

DATASET_FOLDER   = DATASET                       # alias so the old code below works
PROCESSED_FOLDER = PROCESSED
REPORT_FOLDER    = REPORTS / "preprocess"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)


#%% Build the pipeline
# Each arrow in the DAG is inferred automatically: a function's
# parameter name must match some upstream function's `output_name`.
pipeline = Pipeline(
    [
        load_metadata,                 # dataset_folder          -> patient_records
        add_diagnostic_labels,         # patient_records         -> patients_with_diagnosis
        keep_patients_with_recordings, # patients_with_diagnosis -> patients_on_disk
        load_raw_signals,              # patients_on_disk        -> ecg_waveforms
        clean_ecg_signals,             # ecg_waveforms           -> cleaned_ecg_signals
        build_diagnosis_matrix,        # patients_on_disk        -> diagnosis_matrix
        split_by_fold,                 # cleaned + matrix + meta -> dataset_splits
    ],
    profile=True,                      # collect per-node runtime stats
)


#%% Draw the DAG (saved as PNG + shown inline in Spyder's Plots pane)
# Needs the system Graphviz binary + `pip install "pipefunc[all]"`.
try:
    pipeline.visualize_graphviz(filename=REPORT_FOLDER / "pipeline_dag.png")
except Exception as e:
    # If Graphviz is missing, fall back to the Matplotlib renderer.
    print(f"[warn] graphviz export skipped: {e}")
pipeline.visualize()


#%% Execute the pipeline end-to-end
splits = pipeline(
    "dataset_splits",                  # <- positional output name (the leaf)
    dataset_folder=DATASET_FOLDER,
    sampling_rate=100,                 # 100 Hz version of PTB-XL
    val_fold=9,
    test_fold=10,
)


#%% Save the splits so the training script can load them instantly
for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
    np.save(PROCESSED_FOLDER / f"{name}.npy", splits[name])

(PROCESSED_FOLDER / "class_names.txt").write_text(
    "\n".join(splits["class_names"])
)


#%% Print a short human-readable summary
print("\n=== Saved to", PROCESSED_FOLDER, "===")
for name in ("X_train", "X_val", "X_test"):
    print(f"{name:7s} shape = {splits[name].shape}")
for name in ("y_train", "y_val", "y_test"):
    positives = dict(zip(splits["class_names"],
                         splits[name].sum(0).astype(int)))
    print(f"{name:7s} shape = {splits[name].shape}   "
          f"positives per class = {positives}")
print("class order:", splits["class_names"])