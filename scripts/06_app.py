#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — Streamlit demo (browse test set OR upload your own ECG)."""

import sys, io, tempfile
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import torch
import wfdb
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from src.paths import PROCESSED, REPORTS
from src.model import ECGNet

TRAIN_OUT = REPORTS / "train"
FAIRNESS  = REPORTS / "fairness"
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

st.set_page_config(page_title="PTB-XL ECG demo", layout="wide")
st.title("🫀 PTB-XL ECG classifier")

# ---------- cached loaders ----------
@st.cache_resource
def load_model(n_classes):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ECGNet(n_leads=12, n_classes=n_classes).to(dev)
    m.load_state_dict(torch.load(TRAIN_OUT / "best_model.pt", map_location=dev))
    m.eval()
    return m, dev

@st.cache_data
def load_test():
    X = np.load(PROCESSED / "X_test.npy")
    y = np.load(PROCESSED / "y_test.npy")
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    classes = (PROCESSED / "class_names.txt").read_text().splitlines()
    return X, y, classes

def clean_signal(sig_12x1000):
    """Apply the same band-pass + z-score you used in preprocess.py."""
    b, a = butter(3, [0.5, 40], btype="band", fs=100)
    out = np.empty_like(sig_12x1000)
    for k in range(sig_12x1000.shape[1]):
        x = filtfilt(b, a, sig_12x1000[:, k])
        mu, sd = x.mean(), x.std() + 1e-8
        out[:, k] = (x - mu) / sd
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

def predict(model, dev, sig):       # sig: (1000, 12)
    x = torch.from_numpy(sig).float().permute(1, 0).unsqueeze(0).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy().ravel()

X_test, y_test, classes = load_test()
model, device = load_model(len(classes))

# ---------- sidebar ----------
mode = st.sidebar.radio("Input source",
                        ["Browse test set", "Upload WFDB record",
                         "Upload .npy (1000×12)"])
import json

# Load the tuned thresholds, fall back to 0.5 if file missing
thr_path = TRAIN_OUT / "thresholds.json"
if thr_path.exists():
    thr_raw = json.loads(thr_path.read_text())
    default_thresholds = {c: thr_raw[c]["threshold"] for c in classes}
else:
    default_thresholds = {c: 0.5 for c in classes}

st.sidebar.markdown("**Decision thresholds** (pre-set from validation tuning)")
thresholds = {
    c: st.sidebar.slider(c, 0.0, 1.0, float(default_thresholds[c]), 0.01)
    for c in classes
}
# ---------- get a signal + optional ground truth ----------
sig, truth = None, None
if mode == "Browse test set":
    idx = st.sidebar.number_input("test-set index",
                                  0, len(X_test) - 1, 0, 1)
    sig = X_test[idx]
    truth = y_test[idx]

elif mode == "Upload WFDB record":
    st.sidebar.write("Upload both `.hea` and `.dat` files")
    files = st.sidebar.file_uploader(".hea + .dat",
                                     type=["hea", "dat"],
                                     accept_multiple_files=True)
    if files and len(files) == 2:
        with tempfile.TemporaryDirectory() as td:
            for f in files:
                (Path(td) / f.name).write_bytes(f.read())
            stem = Path([f.name for f in files if f.name.endswith(".hea")][0]).stem
            rec = wfdb.rdrecord(str(Path(td) / stem))
            raw = rec.p_signal                                     # (N, 12)
            if rec.fs != 100:                                      # resample
                step = rec.fs // 100
                raw = raw[::step]
            raw = raw[:1000]
            if raw.shape[0] == 1000 and raw.shape[1] == 12:
                sig = clean_signal(raw)

elif mode == "Upload .npy (1000×12)":
    up = st.sidebar.file_uploader("1000×12 float array", type=["npy"])
    if up is not None:
        arr = np.load(io.BytesIO(up.read()))
        if arr.shape == (1000, 12):
            sig = clean_signal(arr)
        else:
            st.error(f"expected shape (1000, 12), got {arr.shape}")

# ---------- show result ----------
if sig is None:
    st.info("Pick a patient or upload a file to see a prediction.")
    st.stop()

probs = predict(model, device, sig)
# new — per-class threshold applied individually
thr_array = np.array([thresholds[c] for c in classes])
preds = (probs >= thr_array).astype(int)

pred_labels = [c for c, v in zip(classes, preds) if v] or ["(none)"]
c1, c2 = st.columns(2)
c2.metric("Predicted", ", ".join(pred_labels))
if truth is not None:
    true_labels = [c for c, v in zip(classes, truth) if v > 0.5] or ["(none)"]
    c1.metric("True", ", ".join(true_labels))

st.subheader("Per-class probability")
st.bar_chart(pd.Series(probs, index=classes))

st.subheader("12-lead ECG")
fig, axes = plt.subplots(6, 2, figsize=(12, 10), sharex=True)
for i, ax in enumerate(axes.T.ravel()):
    ax.plot(sig[:, i], linewidth=0.8)
    ax.set_ylabel(LEADS[i], rotation=0, labelpad=20)
axes[-1, 0].set_xlabel("time (1/100 s)")
axes[-1, 1].set_xlabel("time (1/100 s)")
fig.tight_layout()
st.pyplot(fig)

# ---------- fairness panel ----------
st.header("Fairness audit")
fcols = st.columns(2)
if (FAIRNESS / "auroc_by_sex.png").exists():
    fcols[0].image(str(FAIRNESS / "auroc_by_sex.png"), caption="AUROC by sex")
if (FAIRNESS / "auroc_by_age.png").exists():
    fcols[1].image(str(FAIRNESS / "auroc_by_age.png"), caption="AUROC by age")
if (FAIRNESS / "subgroup_metrics.csv").exists():
    st.dataframe(pd.read_csv(FAIRNESS / "subgroup_metrics.csv"))