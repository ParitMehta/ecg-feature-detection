#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:00:55 2026

@author: mehta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL exploratory checks — written to be readable by clinicians.

The diagram produced at the end reads left-to-right:
    dataset folder
        -> patient records
        -> patients with diagnosis
        -> patients we can actually open
        -> (class balance, unlabeled patients, sample ECGs,
            labels per patient, fold balance, reference ECGs,
            signal quality checks)
        -> overall EDA summary
"""

import sys, os
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

DATASET_FOLDER = "/home/mehta/ML Projects/healthtech/data/physionet.org/files/ptb-xl/1.0.3"
OUT = Path("/home/mehta/ML Projects/healthtech/reports/eda")
OUT.mkdir(parents=True, exist_ok=True)


@pipefunc(output_name="patients_on_disk")
def keep_patients_with_recordings(patients_with_diagnosis, dataset_folder):
    """Keep only patients whose ECG files are actually present."""
    df = patients_with_diagnosis
    ok = df.filename_lr.apply(lambda f: os.path.exists(f"{dataset_folder}/{f}.hea"))
    return df[ok].copy()


@pipefunc(output_name="diagnosis_group_counts")
def count_patients_per_diagnosis(patients_on_disk):
    """How many patients fall into each of the 5 diagnosis groups."""
    counts = patients_on_disk.diagnostic_superclass.explode().value_counts()
    counts.to_csv(OUT / "class_counts.csv", header=["count"])
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Patients per diagnosis group"); ax.set_ylabel("patients")
    fig.tight_layout()
    fig.savefig(OUT / "01_class_counts.png", dpi=150, bbox_inches="tight")
    plt.show()
    return counts


@pipefunc(output_name="patients_without_diagnosis")
def count_patients_without_diagnosis(patients_on_disk):
    """How many patients have no recognised diagnosis attached."""
    n = int(patients_on_disk.diagnostic_superclass.apply(len).eq(0).sum())
    print(f"Patients without any diagnosis: {n}")
    return n


@pipefunc(output_name="first_three_ecgs")
def plot_first_three_ecgs(patients_on_disk, dataset_folder):
    """Plot lead I for the first three patients as a visual check."""
    signals = load_raw_signals.func(
        patients_on_disk.head(5), dataset_folder, sampling_rate=100)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axes[i].plot(signals[i, :, 0]); axes[i].set_ylabel(f"patient {i+1}")
    axes[-1].set_xlabel("time (1/100 s)")
    fig.suptitle("Lead I — first three patients"); fig.tight_layout()
    fig.savefig(OUT / "02_lead_I_first_three.png", dpi=150, bbox_inches="tight")
    plt.show()
    return signals.shape


@pipefunc(output_name="diagnoses_per_patient")
def count_diagnoses_per_patient(patients_on_disk):
    """How many diagnoses each patient carries (0, 1, 2, ...)."""
    lc = (patients_on_disk.diagnostic_superclass.apply(len)
          .value_counts().sort_index())
    lc.to_csv(OUT / "labels_per_record.csv", header=["count"])
    return lc


@pipefunc(output_name="diagnosis_balance_per_fold")
def diagnosis_balance_per_fold(patients_on_disk):
    """Check that each of the 10 data folds has a similar diagnosis mix."""
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
    ax.set_title("Diagnosis mix in each data fold"); ax.set_ylabel("share")
    fig.tight_layout()
    fig.savefig(OUT / "03_per_fold_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return dist


@pipefunc(output_name="reference_ecgs")
def plot_reference_ecgs(patients_on_disk, dataset_folder):
    """Save one example ECG per diagnosis group, plus all 12 leads."""
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
    fig.suptitle("Lead I — one example per diagnosis group"); fig.tight_layout()
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


@pipefunc(output_name="signal_quality_checks")
def signal_quality_checks(patients_on_disk, dataset_folder):
    """Check 100 random ECGs for missing values, strange amplitudes, flat leads."""
    sample = load_raw_signals.func(
        patients_on_disk.sample(100, random_state=0),
        dataset_folder, sampling_rate=100)
    info = dict(
        has_missing_values=bool(np.isnan(sample).any()),
        shape=sample.shape,
        min_voltage_mV=float(sample.min()),
        max_voltage_mV=float(sample.max()),
        quietest_lead_variation=float(sample.std(axis=(0, 1)).min()),
    )
    print("Signal quality:", info)
    return info


@pipefunc(output_name="eda_summary")
def build_summary(diagnosis_group_counts, patients_without_diagnosis,
                  first_three_ecgs, diagnoses_per_patient,
                  diagnosis_balance_per_fold, reference_ecgs,
                  signal_quality_checks):
    """One final summary gathering every EDA result in one place."""
    summary = dict(
        diagnosis_group_counts=diagnosis_group_counts.to_dict(),
        patients_without_diagnosis=patients_without_diagnosis,
        first_three_ecgs_shape=first_three_ecgs,
        diagnoses_per_patient=diagnoses_per_patient.to_dict(),
        fold_balance_shape=diagnosis_balance_per_fold.shape,
        reference_ecgs_saved=reference_ecgs,
        signal_quality=signal_quality_checks,
    )
    (OUT / "eda_summary.txt").write_text(repr(summary))
    return summary


pipeline = Pipeline(
    [
        load_metadata,
        add_diagnostic_labels,
        keep_patients_with_recordings,
        count_patients_per_diagnosis,
        count_patients_without_diagnosis,
        plot_first_three_ecgs,
        count_diagnoses_per_patient,
        diagnosis_balance_per_fold,
        plot_reference_ecgs,
        signal_quality_checks,
        build_summary,
    ],
    profile=True,
)

pipeline.visualize()

summary = pipeline("eda_summary", dataset_folder=DATASET_FOLDER)
print("Done. Files saved to", OUT)