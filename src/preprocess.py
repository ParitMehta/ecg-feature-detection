#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL preprocessing.

Turns what the loader produced into the three things a classifier
needs:

  * cleaned ECG signals   — raw voltages tidied up (no slow baseline
                            drift, consistent voltage scale).
  * diagnosis matrix      — each patient's diagnosis list rewritten as
                            a row of 0/1 values across the 5 groups
                            (NORM, MI, STTC, CD, HYP).
  * dataset splits        — rows divided into training, validation,
                            and test sets using PTB-XL's built-in
                            `strat_fold` column.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MultiLabelBinarizer
from pipefunc import pipefunc


# =============================================================================
# Step 0 — Keep only patients whose ECG files are actually on disk
# -----------------------------------------------------------------------------
# A partial PTB-XL download may be missing some waveform files. This
# node drops any patient whose header file (.hea) is absent, so the
# signal loader cannot later crash on missing files.
# =============================================================================
@pipefunc(output_name="patients_on_disk")
def keep_patients_with_recordings(patients_with_diagnosis, dataset_folder):
    """Keep only patients whose .hea file exists under dataset_folder."""
    df = patients_with_diagnosis
    ok = df.filename_lr.apply(
        lambda f: os.path.exists(f"{dataset_folder}/{f}.hea")
    )
    kept = df[ok].copy()
    print(f"[filter] kept {ok.sum()}/{len(df)} "
          f"({100*ok.sum()/len(df):.1f}%)")
    return kept


# =============================================================================
# Step 1 — Clean the ECG signals
# -----------------------------------------------------------------------------
# Raw ECGs suffer from two common nuisances:
#   (a) "baseline wander" — a slow drift of the whole trace caused by
#       breathing and skin movement (frequencies below ~0.5 Hz).
#   (b) muscle noise and mains interference (above ~40 Hz).
#
# A Butterworth bandpass filter (a standard digital filter that keeps
# a chosen frequency range and suppresses everything outside it)
# removes both. Afterwards each lead is z-scored — subtract the mean,
# divide by the standard deviation — so every patient and every lead
# share the same voltage scale.
# =============================================================================
@pipefunc(output_name="cleaned_ecg_signals")
def clean_ecg_signals(ecg_waveforms: np.ndarray,
                      sampling_rate: int = 100) -> np.ndarray:
    """Remove baseline drift and high-frequency noise, then standardise."""
    low_cut, high_cut = 0.5, 40.0
    nyquist = 0.5 * sampling_rate
    b, a = butter(N=3,
                  Wn=[low_cut / nyquist, high_cut / nyquist],
                  btype="band")
    filtered = filtfilt(b, a, ecg_waveforms, axis=1)
    mean = filtered.mean(axis=1, keepdims=True)
    std  = filtered.std(axis=1, keepdims=True) + 1e-8
    return ((filtered - mean) / std).astype(np.float32)


# =============================================================================
# Step 2 — Turn the diagnosis lists into a 0/1 matrix
# -----------------------------------------------------------------------------
# The loader gives every patient a Python list of diagnosis names,
# e.g. ["MI", "STTC"]. A classifier needs a rectangular table of
# numbers instead. MultiLabelBinarizer does that: each diagnosis group
# becomes one column, each patient becomes one row of 0s and 1s.
# =============================================================================
@pipefunc(output_name="diagnosis_matrix")
def build_diagnosis_matrix(patients_on_disk: pd.DataFrame):
    """Return dict(labels=0/1 array, class_names=list of group names)."""
    groups = ["NORM", "MI", "STTC", "CD", "HYP"]
    mlb = MultiLabelBinarizer(classes=groups)
    labels = mlb.fit_transform(patients_on_disk.diagnostic_superclass)
    return dict(labels=labels.astype(np.float32),
                class_names=list(mlb.classes_))


# =============================================================================
# Step 3 — Split into training, validation, and test sets
# -----------------------------------------------------------------------------
# PTB-XL ships with a column `strat_fold` (values 1-10) that already
# spreads the diagnoses evenly across ten groups. The PTB-XL paper
# recommends:
#   folds 1-8  -> training    (what the model learns from)
#   fold  9    -> validation  (used to tune settings; not for final score)
#   fold  10   -> test        (used exactly once at the very end)
# Using this convention keeps results comparable with the published
# PTB-XL benchmarks.
# =============================================================================
@pipefunc(output_name="dataset_splits")
def split_by_fold(cleaned_ecg_signals: np.ndarray,
                  diagnosis_matrix: dict,
                  patients_on_disk: pd.DataFrame,
                  val_fold: int = 9,
                  test_fold: int = 10) -> dict:
    """Partition signals and labels into train/val/test by strat_fold."""
    labels = diagnosis_matrix["labels"]
    folds  = patients_on_disk.strat_fold.to_numpy()

    train_mask = ~np.isin(folds, [val_fold, test_fold])
    val_mask   = folds == val_fold
    test_mask  = folds == test_fold

    splits = dict(
        X_train=cleaned_ecg_signals[train_mask],
        y_train=labels[train_mask],
        X_val  =cleaned_ecg_signals[val_mask],
        y_val  =labels[val_mask],
        X_test =cleaned_ecg_signals[test_mask],
        y_test =labels[test_mask],
        class_names=diagnosis_matrix["class_names"],
    )
    print(f"[split] train={train_mask.sum()} "
          f"val={val_mask.sum()} test={test_mask.sum()}")
    return splits