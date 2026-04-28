

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — inspect the trained model's predictions.

Prerequisite: scripts/03_train.py has been run at least once
(writes reports/train/best_model.pt).
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

from src.model import ECGNet

from src.paths import PROCESSED, REPORTS

TRAIN_OUT = REPORTS / "train"
OUT       = REPORTS / "inspect"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5   # probability above which we say "yes, this diagnosis"


def main():
    # ---- Load test split ----
    X_test = np.load(PROCESSED / "X_test.npy")
    y_test = np.load(PROCESSED / "y_test.npy")
    class_names = (PROCESSED / "class_names.txt").read_text().splitlines()

    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    X_t = torch.from_numpy(X_test).float().permute(0, 2, 1)   # (N, 12, 1000)

    # ---- Load the trained model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(n_leads=12, n_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(TRAIN_OUT / "best_model.pt",
                                     map_location=device))
    model.eval()
    print("Loaded best model from", TRAIN_OUT / "best_model.pt")

    # ---- Run predictions in batches (keeps memory reasonable) ----
    probs = np.zeros((len(X_t), len(class_names)), dtype=np.float32)
    with torch.no_grad():
        bs = 128
        for i in range(0, len(X_t), bs):
            batch = X_t[i:i + bs].to(device)
            probs[i:i + bs] = torch.sigmoid(model(batch)).cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)

    # ---- Per-class AUROC ----
    aurocs = [roc_auc_score(y_test[:, k], probs[:, k])
              for k in range(len(class_names))]
    per_class = dict(zip(class_names, aurocs))
    print("per-class test AUROC:", per_class)
    print("macro AUROC:        ", float(np.mean(aurocs)))

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, aurocs, color="steelblue")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Test-set AUROC per diagnosis group")
    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "per_class_auroc.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- CSV of first 20 patients ----
    rows = []
    for i in range(20):
        rows.append({
            "patient_index": i,
            "true":      [c for c, y in zip(class_names, y_test[i]) if y > 0.5],
            "predicted": [c for c, p in zip(class_names, preds[i])  if p > 0.5],
            **{f"prob_{c}": round(float(probs[i, k]), 3)
               for k, c in enumerate(class_names)},
        })
    pd.DataFrame(rows).to_csv(OUT / "predictions_head.csv", index=False)

    # ---- Gallery: 3 correct + 3 incorrect example patients ----
    def labels_to_text(vec):
        picks = [c for c, y in zip(class_names, vec) if y > 0.5]
        return ", ".join(picks) if picks else "(none)"

    exact_match = np.all(preds == y_test.astype(int), axis=1)
    agree_idx    = np.where(exact_match)[0][:3].tolist()
    disagree_idx = np.where(~exact_match)[0][:3].tolist()
    picks = agree_idx + disagree_idx

    for n, idx in enumerate(picks, 1):
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(X_test[idx, :, 1])   # lead II
        ax.set_title(
            f"patient #{idx}   "
            f"true: {labels_to_text(y_test[idx])}   |   "
            f"predicted: {labels_to_text(preds[idx])}"
        )
        ax.set_xlabel("time (1/100 s)")
        ax.set_ylabel("lead II (z-scored)")
        fig.tight_layout()
        fig.savefig(OUT / f"patient_{n:02d}_idx{idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print("Done. Artefacts saved to", OUT)


if __name__ == "__main__":
    main()