#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — Exploratory Data Analysis pipeline.

Reuses the ingestion nodes from src/loader.py:

    load_metadata             -> patient_records
    add_diagnostic_labels     -> patients_with_diagnosis
    load_raw_signals          -> ecg_waveforms

and the filter node from src/preprocess.py:

    keep_patients_with_recordings -> patients_on_disk
"""

#%% Imports and project paths
import sys
from pathlib import Path
from itertools import chain

sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipefunc import pipefunc, Pipeline

from src.loader import (
    load_metadata, add_diagnostic_labels, load_raw_signals,
)
from src.preprocess import keep_patients_with_recordings

from src.paths import DATASET


DATASET_FOLDER = "/home/mehta/ML Projects/healthtech/data/physionet.org/files/ptb-xl/1.0.3"
OUT = Path("/home/mehta/ML Projects/healthtech/reports/eda")
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Stage A — How many patients per diagnosis group
# =============================================================================
@pipefunc(output_name="diagnosis_group_counts")
def class_counts(patients_on_disk):
    """Bar chart of patients per diagnosis group."""
    counts = patients_on_disk.diagnostic_superclass.explode().value_counts()
    counts.to_csv(OUT / "class_counts.csv", header=["count"])
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Patients per diagnosis group"); ax.set_ylabel("patients")
    fig.tight_layout()
    fig.savefig(OUT / "01_class_counts.png", dpi=150, bbox_inches="tight")
    plt.show()
    return counts


# =============================================================================
# Stage B — Patients without any diagnosis
# =============================================================================
@pipefunc(output_name="patients_without_diagnosis")
def unlabeled(patients_on_disk):
    """Count patients whose code list maps to zero diagnosis groups."""
    n = int(patients_on_disk.diagnostic_superclass.apply(len).eq(0).sum())
    print(f"[unlabeled] {n} of {len(patients_on_disk)} patients")
    return n


# =============================================================================
# Stage C — Visual smoke test: lead I of the first three patients
# =============================================================================
@pipefunc(output_name="first_three_ecgs")
def preview_signals(patients_on_disk, dataset_folder):
    """Plot lead I of the first three patients."""
    sig = load_raw_signals.func(
        patients_on_disk.head(5), dataset_folder, sampling_rate=100)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axes[i].plot(sig[i, :, 0]); axes[i].set_ylabel(f"patient {i+1}")
    axes[-1].set_xlabel("time (1/100 s)")
    fig.suptitle("Lead I — first three patients"); fig.tight_layout()
    fig.savefig(OUT / "02_lead_I_first_three.png", dpi=150, bbox_inches="tight")
    plt.show()
    return sig.shape


# =============================================================================
# Stage D — How many diagnoses each patient carries
# =============================================================================
@pipefunc(output_name="diagnoses_per_patient")
def label_cardinality(patients_on_disk):
    """Distribution of number-of-diagnoses per patient."""
    lc = (patients_on_disk.diagnostic_superclass.apply(len)
          .value_counts().sort_index())
    lc.to_csv(OUT / "labels_per_record.csv", header=["count"])
    return lc


# =============================================================================
# Stage E — Diagnosis balance across the 10 PTB-XL folds
# =============================================================================
@pipefunc(output_name="fold_balance")
def fold_distribution(patients_on_disk):
    """Stacked bar of diagnosis proportions per strat_fold."""
    rows = []
    for fold in range(1, 11):
        sub = patients_on_disk[patients_on_disk.strat_fold == fold]
        flat = list(chain.from_iterable(sub.diagnostic_superclass))
        c = pd.Series(flat).value_counts(normalize=True)
        c.name = f"fold {fold}"
        rows.append(c)
    dist = pd.concat(rows, axis=1).fillna(0)
    dist.to_csv(OUT / "per_fold_distribution.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    dist.T.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Diagnosis mix per fold"); ax.set_ylabel("share")
    fig.tight_layout()
    fig.savefig(OUT / "03_per_fold_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return dist


# =============================================================================
# Stage F — Reference ECGs: one per diagnosis group + all 12 leads
# =============================================================================
@pipefunc(output_name="reference_ecgs")
def reference_examples(patients_on_disk, dataset_folder):
    """Save per-group and 12-lead reference figures."""
    groups = ["NORM", "MI", "STTC", "CD", "HYP"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for ax, g in zip(axes, groups):
        idx = patients_on_disk[
            patients_on_disk.diagnostic_superclass.apply(lambda l: l == [g])
        ].index[0]
        sig = load_raw_signals.func(
            patients_on_disk.loc[[idx]], dataset_folder, sampling_rate=100)
        ax.plot(sig[0, :, 0]); ax.set_ylabel(g)
    axes[-1].set_xlabel("time (1/100 s)")
    fig.suptitle("Lead I — one patient per diagnosis group"); fig.tight_layout()
    fig.savefig(OUT / "04_one_record_per_superclass.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    sig = load_raw_signals.func(patients_on_disk.head(1),
                                dataset_folder, sampling_rate=100)
    fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, leads)):
        ax.plot(sig[0, :, i]); ax.set_ylabel(name)
    fig.suptitle("All 12 leads — first patient"); fig.tight_layout()
    fig.savefig(OUT / "05_all_12_leads.png", dpi=150, bbox_inches="tight")
    plt.show()
    return True


# =============================================================================
# Stage G — Signal quality checks on 100 random ECGs
# =============================================================================
@pipefunc(output_name="signal_quality")
def sanity(patients_on_disk, dataset_folder):
    """Check 100 random ECGs for NaNs, bad amplitudes, dead leads."""
    s = load_raw_signals.func(
        patients_on_disk.sample(100, random_state=0),
        dataset_folder, sampling_rate=100)
    info = dict(
        has_missing_values=bool(np.isnan(s).any()),
        shape=s.shape,
        min_voltage_mV=float(s.min()),
        max_voltage_mV=float(s.max()),
        quietest_lead_variation=float(s.std(axis=(0, 1)).min()),
    )
    print("[signal_quality]", info)
    return info


# =============================================================================
# Stage H — Final single-leaf summary
# =============================================================================
@pipefunc(output_name="eda_summary")
def assemble_report(diagnosis_group_counts, patients_without_diagnosis,
                    first_three_ecgs, diagnoses_per_patient,
                    fold_balance, reference_ecgs, signal_quality):
    """Gather every EDA result into one dictionary."""
    report = dict(
        diagnosis_group_counts=diagnosis_group_counts.to_dict(),
        patients_without_diagnosis=patients_without_diagnosis,
        first_three_ecgs_shape=first_three_ecgs,
        diagnoses_per_patient=diagnoses_per_patient.to_dict(),
        fold_balance_shape=fold_balance.shape,
        reference_ecgs_saved=reference_ecgs,
        signal_quality=signal_quality,
    )
    (OUT / "eda_summary.txt").write_text(repr(report))
    return report


#%% Build, visualize, run
pipeline = Pipeline(
    [
        load_metadata,
        add_diagnostic_labels,
        keep_patients_with_recordings,
        class_counts,
        unlabeled,
        preview_signals,
        label_cardinality,
        fold_distribution,
        reference_examples,
        sanity,
        assemble_report,
    ],
    profile=True,
)

pipeline.visualize()
report = pipeline("eda_summary", dataset_folder=DATASET_FOLDER)
print("Done. Outputs saved to", OUT)