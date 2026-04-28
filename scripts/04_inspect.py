#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — inspect the trained model's predictions."""

import sys, json
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    multilabel_confusion_matrix,
)

from src.model import ECGNet
from src.paths import PROCESSED, REPORTS

TRAIN_OUT = REPORTS / "train"
OUT = REPORTS / "inspect"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    X_test = np.load(PROCESSED / "X_test.npy")
    y_test = np.load(PROCESSED / "y_test.npy").astype(int)
    class_names = (PROCESSED / "class_names.txt").read_text().splitlines()
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    X_t = torch.from_numpy(X_test).float().permute(0, 2, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(n_leads=12, n_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(TRAIN_OUT / "best_model.pt",
                                     map_location=device))
    model.eval()
    print("Loaded best model from", TRAIN_OUT / "best_model.pt")

    # ---- Load per-class thresholds (fallback to 0.5) ----
    thr_path = TRAIN_OUT / "thresholds.json"
    if thr_path.exists():
        thr_raw = json.loads(thr_path.read_text())
        thresholds = np.array([thr_raw[c]["threshold"] for c in class_names])
        print("Using val-tuned thresholds:",
              dict(zip(class_names, thresholds.tolist())))
    else:
        thresholds = np.full(len(class_names), 0.5)
        print("thresholds.json not found — falling back to 0.5 for all classes")

    # ---- Predictions ----
    probs = np.zeros((len(X_t), len(class_names)), dtype=np.float32)
    with torch.no_grad():
        bs = 128
        for i in range(0, len(X_t), bs):
            probs[i:i + bs] = torch.sigmoid(
                model(X_t[i:i + bs].to(device))
            ).cpu().numpy()
    preds = (probs >= thresholds).astype(int)

    # ---- Per-class metrics ----
    rows = []
    for k, c in enumerate(class_names):
        rows.append({
            "class": c,
            "threshold": float(thresholds[k]),
            "auroc":     round(roc_auc_score(y_test[:, k], probs[:, k]), 4),
            "f1":        round(f1_score(y_test[:, k], preds[:, k], zero_division=0), 4),
            "precision": round(precision_score(y_test[:, k], preds[:, k], zero_division=0), 4),
            "recall":    round(recall_score(y_test[:, k], preds[:, k], zero_division=0), 4),
            "support":   int(y_test[:, k].sum()),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.loc[len(metrics_df)] = {
        "class": "macro",
        "threshold": np.nan,
        "auroc":     round(metrics_df["auroc"].mean(), 4),
        "f1":        round(metrics_df["f1"].mean(), 4),
        "precision": round(metrics_df["precision"].mean(), 4),
        "recall":    round(metrics_df["recall"].mean(), 4),
        "support":   int(y_test.sum()),
    }
    metrics_df.to_csv(OUT / "metrics_per_class.csv", index=False)
    print(metrics_df.to_string(index=False))

    # ---- AUROC bar chart ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, metrics_df.iloc[:-1]["auroc"], color="steelblue")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("Test-set AUROC per diagnosis group")
    for i, v in enumerate(metrics_df.iloc[:-1]["auroc"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "per_class_auroc.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Confusion matrices (one 2x2 per class) ----
    cms = multilabel_confusion_matrix(y_test, preds)
    fig, axes = plt.subplots(1, len(class_names),
                             figsize=(3.2 * len(class_names), 3.2))
    for ax, cm, c in zip(axes, cms, class_names):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(c)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred neg", "pred pos"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true neg", "true pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Per-class confusion matrices (test set, val-tuned thresholds)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Label co-occurrence in predictions (diagnosis for NORM+positive issue) ----
    cooc = preds.T @ preds  # (C, C)
    cooc_df = pd.DataFrame(cooc, index=class_names, columns=class_names)
    cooc_df.to_csv(OUT / "pred_cooccurrence.csv")

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

    # ---- Gallery ----
    def labels_to_text(vec):
        picks = [c for c, y in zip(class_names, vec) if y > 0.5]
        return ", ".join(picks) if picks else "(none)"

    exact_match = np.all(preds == y_test, axis=1)
    picks = np.where(exact_match)[0][:3].tolist() + \
            np.where(~exact_match)[0][:3].tolist()

    for n, idx in enumerate(picks, 1):
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(X_test[idx, :, 1])
        ax.set_title(
            f"patient #{idx}  true: {labels_to_text(y_test[idx])} | "
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