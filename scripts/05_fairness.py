#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — fairness audit by sex and age.

Prerequisite: scripts/03_train.py has run at least once
(writes reports/train/best_model.pt).

Outputs to reports/fairness/:
    auroc_by_sex.png
    auroc_by_age.png
    subgroup_metrics.csv
"""

#%% Imports and project paths
import sys
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.loader import load_metadata, add_diagnostic_labels, load_raw_signals
from src.preprocess import keep_patients_with_recordings
from src.model import ECGNet

from src.paths import DATASET, PROCESSED, REPORTS

TRAIN_OUT = REPORTS / "train"
OUT       = REPORTS / "fairness"
OUT.mkdir(parents=True, exist_ok=True)


def subgroup_auroc(y_true, y_prob, mask, class_names):
    """Per-class + macro AUROC on the subset where mask is True."""
    y = y_true[mask]; p = y_prob[mask]
    per_class = {}
    for k, c in enumerate(class_names):
        if 0 < y[:, k].sum() < len(y):
            per_class[c] = roc_auc_score(y[:, k], p[:, k])
        else:
            per_class[c] = np.nan
    macro = float(np.nanmean(list(per_class.values())))
    return macro, per_class, int(mask.sum())


def main():
    # ---- Load test split ----
    X_test = np.load(PROCESSED / "X_test.npy")
    y_test = np.load(PROCESSED / "y_test.npy")
    class_names = (PROCESSED / "class_names.txt").read_text().splitlines()
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Reload demographics for test patients ----
    # Re-run the minimal metadata pipeline and use strat_fold == 10 as "test"
    # (this mirrors what 02_preprocess.py did to build y_test).
    meta = load_metadata.func(DATASET)
    meta = add_diagnostic_labels.func(meta, DATASET)
    meta = keep_patients_with_recordings.func(meta, DATASET)
    test_meta = meta[meta.strat_fold == 10].reset_index(drop=True)

    assert len(test_meta) == len(X_test), (
        f"metadata rows ({len(test_meta)}) != X_test rows ({len(X_test)})"
    )

    # ---- Run predictions on the whole test set ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(n_leads=12, n_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(TRAIN_OUT / "best_model.pt",
                                     map_location=device))
    model.eval()

    X_t = torch.from_numpy(X_test).float().permute(0, 2, 1)
    probs = np.zeros((len(X_t), len(class_names)), dtype=np.float32)
    with torch.no_grad():
        bs = 128
        for i in range(0, len(X_t), bs):
            probs[i:i + bs] = torch.sigmoid(
                model(X_t[i:i + bs].to(device))
            ).cpu().numpy()

    # ---- Define subgroups ----
    sex = test_meta["sex"].to_numpy()       # 0 = male, 1 = female in PTB-XL
    age = test_meta["age"].to_numpy()
    age_bin = pd.cut(
        age, bins=[-0.1, 40, 60, 75, 200],
        labels=["<40", "40-60", "60-75", "75+"],
    ).astype(str)

    groups = {
        "male":    sex == 0,
        "female":  sex == 1,
        "age <40":   age_bin == "<40",
        "age 40-60": age_bin == "40-60",
        "age 60-75": age_bin == "60-75",
        "age 75+":   age_bin == "75+",
    }

    # ---- Compute metrics ----
    rows = []
    for name, mask in groups.items():
        macro, per_class, n = subgroup_auroc(y_test, probs, mask, class_names)
        row = {"group": name, "n": n, "macro_auroc": round(macro, 3)}
        for c, v in per_class.items():
            row[f"auroc_{c}"] = round(v, 3) if not np.isnan(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subgroup_metrics.csv", index=False)
    print(df.to_string(index=False))

    # ---- Bar chart: macro-AUROC by sex ----
    fig, ax = plt.subplots(figsize=(5, 4))
    sex_rows = df[df.group.isin(["male", "female"])]
    ax.bar(sex_rows.group, sex_rows.macro_auroc, color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("macro-AUROC")
    ax.set_title("Performance by sex (test set)")
    for i, v in enumerate(sex_rows.macro_auroc):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "auroc_by_sex.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Bar chart: macro-AUROC by age group ----
    fig, ax = plt.subplots(figsize=(6, 4))
    age_rows = df[df.group.str.startswith("age ")]
    ax.bar(age_rows.group, age_rows.macro_auroc, color="steelblue")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("macro-AUROC")
    ax.set_title("Performance by age group (test set)")
    for i, v in enumerate(age_rows.macro_auroc):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "auroc_by_age.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Done. Artefacts saved to", OUT)


if __name__ == "__main__":
    main()