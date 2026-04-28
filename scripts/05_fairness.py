#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — fairness audit by sex and age."""

import sys, json
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
)

from src.loader import load_metadata, add_diagnostic_labels, load_raw_signals
from src.preprocess import keep_patients_with_recordings
from src.model import ECGNet
from src.paths import DATASET, PROCESSED, REPORTS

TRAIN_OUT = REPORTS / "train"
OUT = REPORTS / "fairness"
OUT.mkdir(parents=True, exist_ok=True)


def subgroup_metrics(y_true, y_prob, y_pred, mask, class_names):
    """Per-class AUROC / F1 / precision / recall on the masked subset."""
    y = y_true[mask]; p = y_prob[mask]; yh = y_pred[mask]
    out = {}
    for k, c in enumerate(class_names):
        pos = y[:, k].sum()
        if 0 < pos < len(y):
            out[c] = {
                "auroc":     roc_auc_score(y[:, k], p[:, k]),
                "f1":        f1_score(y[:, k], yh[:, k], zero_division=0),
                "precision": precision_score(y[:, k], yh[:, k], zero_division=0),
                "recall":    recall_score(y[:, k], yh[:, k], zero_division=0),
            }
        else:
            out[c] = {"auroc": np.nan, "f1": np.nan,
                      "precision": np.nan, "recall": np.nan}
    macro = {m: float(np.nanmean([out[c][m] for c in class_names]))
             for m in ("auroc", "f1", "precision", "recall")}
    return macro, out, int(mask.sum())


def main():
    X_test = np.load(PROCESSED / "X_test.npy")
    y_test = np.load(PROCESSED / "y_test.npy").astype(int)
    class_names = (PROCESSED / "class_names.txt").read_text().splitlines()
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    meta = load_metadata.func(DATASET)
    meta = add_diagnostic_labels.func(meta, DATASET)
    meta = keep_patients_with_recordings.func(meta, DATASET)
    test_meta = meta[meta.strat_fold == 10].reset_index(drop=True)
    assert len(test_meta) == len(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(n_leads=12, n_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(TRAIN_OUT / "best_model.pt",
                                     map_location=device))
    model.eval()

    # ---- Load tuned thresholds ----
    thr_path = TRAIN_OUT / "thresholds.json"
    if thr_path.exists():
        thr_raw = json.loads(thr_path.read_text())
        thresholds = np.array([thr_raw[c]["threshold"] for c in class_names])
    else:
        thresholds = np.full(len(class_names), 0.5)

    X_t = torch.from_numpy(X_test).float().permute(0, 2, 1)
    probs = np.zeros((len(X_t), len(class_names)), dtype=np.float32)
    with torch.no_grad():
        bs = 128
        for i in range(0, len(X_t), bs):
            probs[i:i + bs] = torch.sigmoid(
                model(X_t[i:i + bs].to(device))
            ).cpu().numpy()
    preds = (probs >= thresholds).astype(int)

    sex = test_meta["sex"].to_numpy()
    age = test_meta["age"].to_numpy()
    age_bin = pd.cut(age, bins=[-0.1, 40, 60, 75, 200],
                     labels=["<40", "40-60", "60-75", "75+"]).astype(str)

    groups = {
        "male":      sex == 0,
        "female":    sex == 1,
        "age <40":   age_bin == "<40",
        "age 40-60": age_bin == "40-60",
        "age 60-75": age_bin == "60-75",
        "age 75+":   age_bin == "75+",
    }

    rows = []
    for name, mask in groups.items():
        macro, per_class, n = subgroup_metrics(y_test, probs, preds,
                                               mask, class_names)
        row = {"group": name, "n": n,
               "macro_auroc":     round(macro["auroc"], 3),
               "macro_f1":        round(macro["f1"], 3),
               "macro_precision": round(macro["precision"], 3),
               "macro_recall":    round(macro["recall"], 3)}
        for c in class_names:
            for m in ("auroc", "f1", "precision", "recall"):
                v = per_class[c][m]
                row[f"{m}_{c}"] = round(v, 3) if not np.isnan(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subgroup_metrics.csv", index=False)
    print(df[["group", "n", "macro_auroc", "macro_f1",
              "macro_precision", "macro_recall"]].to_string(index=False))

    # ---- Bar charts (AUROC + F1) by sex and age ----
    def bar(ax, sub, metric, title, colors):
        ax.bar(sub.group, sub[metric], color=colors)
        ax.set_ylim(0.0, 1.0); ax.set_ylabel(metric)
        ax.set_title(title)
        for i, v in enumerate(sub[metric]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sex_rows = df[df.group.isin(["male", "female"])]
    bar(axes[0], sex_rows, "macro_auroc", "AUROC by sex",
        ["#4C72B0", "#DD8452"])
    bar(axes[1], sex_rows, "macro_f1", "F1 by sex",
        ["#4C72B0", "#DD8452"])
    fig.tight_layout()
    fig.savefig(OUT / "metrics_by_sex.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    age_rows = df[df.group.str.startswith("age ")]
    bar(axes[0], age_rows, "macro_auroc", "AUROC by age group", "steelblue")
    bar(axes[1], age_rows, "macro_f1",    "F1 by age group",    "steelblue")
    fig.tight_layout()
    fig.savefig(OUT / "metrics_by_age.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Done. Artefacts saved to", OUT)


if __name__ == "__main__":
    main()