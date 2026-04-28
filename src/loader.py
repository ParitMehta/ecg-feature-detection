#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL data ingestion.

Three steps, each wired into the project pipeline:

    load_metadata             -> patient_records
    add_diagnostic_labels     -> patients_with_diagnosis
    load_raw_signals          -> ecg_waveforms

These names match the parameters expected by src/preprocess.py, so
pipefunc connects the nodes automatically.
"""

from pathlib import Path
import ast
import numpy as np
import pandas as pd
import wfdb
from pipefunc import pipefunc


# =============================================================================
# Step 1 — Read the PTB-XL patient index
# -----------------------------------------------------------------------------
# ptbxl_database.csv is a big table with one row per ECG recording
# (patient id, filenames, diagnostic codes, fold number, etc.).
# The diagnostic codes are stored as text that LOOKS like a dict
# (e.g. "{'NORM': 100.0}"), so ast.literal_eval turns it into a real
# Python dict that later steps can use.
# =============================================================================
@pipefunc(output_name="patient_records")
def load_metadata(dataset_folder):
    """Open the PTB-XL patient index (one row per ECG recording)."""
    p = Path(dataset_folder)
    df = pd.read_csv(p / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


# =============================================================================
# Step 2 — Attach the 5 broad diagnosis groups
# -----------------------------------------------------------------------------
# PTB-XL has ~70 specific diagnostic codes (IMI, AFIB, ...). Every
# diagnostic code belongs to one of 5 broad groups (superclasses):
# NORM, MI, STTC, CD, HYP. scp_statements.csv is the lookup table
# that tells us which specific code maps to which broad group.
# We only keep rows marked `diagnostic == 1` (real diagnoses, not
# rhythm/annotation notes). A new column `diagnostic_superclass`
# is added, holding the LIST of broad groups that apply to a patient.
# =============================================================================
@pipefunc(output_name="patients_with_diagnosis")
def add_diagnostic_labels(patient_records, dataset_folder):
    """Attach the 5 broad diagnosis groups to every patient record."""
    p = Path(dataset_folder)
    agg = pd.read_csv(p / "scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]

    def to_groups(codes):
        # Set-comprehension: unique broad groups for the codes we know.
        return list({agg.loc[k].diagnostic_class
                     for k in codes if k in agg.index})

    # .copy() keeps this function "pure" — it does not silently change
    # the caller's DataFrame.
    df = patient_records.copy()
    df["diagnostic_superclass"] = df.scp_codes.apply(to_groups)
    return df


# =============================================================================
# Step 3 — Read the actual ECG voltage recordings
# -----------------------------------------------------------------------------
# Each ECG is stored as two files: a header (.hea, text) and the raw
# samples (.dat, binary). PTB-XL provides two resolutions:
#   100 Hz ("lr", low-resolution) — fast to load
#   500 Hz ("hr", high-resolution) — five times larger
# wfdb.rdsamp returns (signal_array, metadata_dict); we only keep the
# signal. Stacked together, the result is shape
# (n_patients, n_time_samples, 12_leads).
#
# This node consumes `patients_on_disk` (the filtered table produced
# by src/preprocess.py) so the signal array is row-aligned with the
# labels downstream.
# =============================================================================
@pipefunc(output_name="ecg_waveforms")
def load_raw_signals(patients_on_disk, dataset_folder, sampling_rate=100):
    """Read the actual ECG voltage recordings for every patient."""
    p = Path(dataset_folder)
    files = (patients_on_disk.filename_lr
             if sampling_rate == 100
             else patients_on_disk.filename_hr)
    return np.array([wfdb.rdsamp(str(p / f))[0] for f in files])