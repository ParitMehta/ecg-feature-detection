#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — Streamlit demo (browse test set OR upload your own ECG)."""

import sys, io, tempfile, json
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import pandas as pd
import wfdb
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import requests

from src.paths import PROCESSED, REPORTS

FAIRNESS = REPORTS / "fairness"
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# URL of your FastAPI backend
API_URL = "http://localhost:8000/predict"  # change to your server URL if needed

st.set_page_config(page_title="ECG Classifier Demo", layout="wide")
st.title("🫀 ECG Diagnostic Classifier")
st.caption("New here? Open the **Overview** page in the sidebar for background on the project, the data, and how to read the results.")

st.markdown(
    """
**On this page you can:**

- Choose or upload an Electro Cardiogram (ECG) (left panel)  
- See the the model’s predicted diagnosis  
- See how confident the model is for each diagnosis  
- See how well the model works for different patient groups 
"""
)
st.markdown("---")

# ── How to use this app ───────────────────────────────────────────────────────
with st.expander("▶  How to use this app", expanded=False):
    st.markdown("""
    1. **Choose an ECG** using the panel on the left.  
       - *Browse test set* — pick from ~2,200 real ECGs the model has never trained on. The correct diagnosis is shown so you can compare.  
       - *Upload WFDB record* — upload a `.hea` + `.dat` file pair from the PTB-XL database.  
       - *Upload .npy* — upload a NumPy file containing a 1000×12 signal table.

    2. **Read the result** at the top of the page: the model's suggested diagnosis and, when available, the confirmed diagnosis.

    3. **Look at the confidence chart** to see how sure the model is about each possible diagnosis.

    4. **Adjust the sensitivity sliders** in the sidebar to see how more or less cautious the model becomes.

    5. **Scroll down** to see the ECG tracing and how the model performs across different patient groups.
    """)

# ---------- cached loaders ----------
@st.cache_data
def load_test():
    X = np.load(PROCESSED / "X_test.npy")
    y = np.load(PROCESSED / "y_test.npy")
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    classes = (PROCESSED / "class_names.txt").read_text().splitlines()
    return X, y, classes

def clean_signal(sig_12x1000):
    b, a = butter(3, [0.5, 40], btype="band", fs=100)
    out = np.empty_like(sig_12x1000)
    for k in range(sig_12x1000.shape[1]):
        x = filtfilt(b, a, sig_12x1000[:, k])
        mu, sd = x.mean(), x.std() + 1e-8
        out[:, k] = (x - mu) / sd
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

def call_backend(sig_12x1000: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Send ECG to FastAPI backend and get probabilities + class names."""
    payload = {"signal": sig_12x1000.tolist()}
    try:
        res = requests.post(API_URL, json=payload, timeout=5)
        res.raise_for_status()
    except Exception as e:
        st.error(f"Error contacting model API: {e}")
        st.stop()
    data = res.json()
    probs = np.array(data.get("probabilities", []), dtype=float)
    classes_from_api = data.get("classes", [])
    return probs, classes_from_api

X_test, y_test, classes = load_test()

# ---------- sidebar ----------
st.sidebar.title("Controls")

st.sidebar.markdown("### Step 1 — Choose an ECG")
st.sidebar.caption(
    "You can either pick a preloaded ECG from the PTB‑XL **test set** "
    "(real patients the model has never trained on), or upload your own file."
)
mode = st.sidebar.radio(
    "Source of the ECG",
    ["Browse test set (preloaded)", "Upload WFDB record", "Upload .npy (1000×12)"]
)

# Load tuned thresholds, fall back to 0.5 if file missing
thr_path = REPORTS / "train" / "thresholds.json"
if thr_path.exists():
    thr_raw = json.loads(thr_path.read_text())
    default_thresholds = {c: thr_raw[c]["threshold"] for c in classes}
else:
    default_thresholds = {c: 0.5 for c in classes}

# ---------- get a signal + optional ground truth ----------
sig, truth = None, None
if mode == "Browse test set (preloaded)":
    idx = st.sidebar.number_input(
        "Pick a patient from the test set (0 to {})".format(len(X_test) - 1),
        0, len(X_test) - 1, 0, 1
    )
    sig   = X_test[idx]
    truth = y_test[idx]

elif mode == "Upload WFDB record":
    st.sidebar.write("Upload both the `.hea` (header) and `.dat` (signal) files together.")
    files = st.sidebar.file_uploader(
        "Select .hea and .dat files",
        type=["hea", "dat"],
        accept_multiple_files=True
    )
    if files and len(files) == 2:
        with tempfile.TemporaryDirectory() as td:
            for f in files:
                (Path(td) / f.name).write_bytes(f.read())
            stem = Path([f.name for f in files if f.name.endswith(".hea")][0]).stem
            rec  = wfdb.rdrecord(str(Path(td) / stem))
            raw  = rec.p_signal
            if rec.fs != 100:
                step = rec.fs // 100
                raw  = raw[::step]
            raw = raw[:1000]
            if raw.shape[0] == 1000 and raw.shape[1] == 12:
                sig = clean_signal(raw)

elif mode == "Upload .npy (1000×12)":
    st.sidebar.write("Upload a NumPy file (.npy) containing a table of 1,000 time steps × 12 leads.")
    up = st.sidebar.file_uploader("Select .npy file", type=["npy"])
    if up is not None:
        arr = np.load(io.BytesIO(up.read()))
        if arr.shape == (1000, 12):
            sig = clean_signal(arr)
        else:
            st.error(
                f"The file has the wrong shape ({arr.shape}). "
                "It must be exactly 1,000 rows (time steps) × 12 columns (leads)."
            )

st.sidebar.markdown("---")
st.sidebar.markdown("### Step 2 — Setting the bar for the diagnosis prediction")

st.sidebar.markdown(
    "For each diagnosis below, you can set **how sure the model must feel before it "
    "speaks up** and flags that diagnosis.\n\n"
    "- A **lower setting** means the model is allowed to speak up earlier (more sensitive): "
    "it will catch more possible cases, but it will also raise more false alarms.\n"
    "- A **higher setting** means the model stays quiet unless it is very sure "
    "(more specific): fewer false alarms, but more missed cases.\n\n"
    "The starting settings are not arbitrary: they come from a separate validation step "
    "where we tested many settings and chose the ones that best balanced missed cases vs "
    "false alarms for each diagnosis. For most uses, it is best to **leave these defaults "
    "as they are** and only move them if you want to explore what would happen if the "
    "model were made more sensitive or more cautious."
)
thresholds = {
    c: st.sidebar.slider(c, 0.0, 1.0, float(default_thresholds[c]), 0.01)
    for c in classes
}

# ---------- show result ----------
if sig is None:
    st.info("👈  Choose a patient or upload an ECG file using the panel on the left to see a result.")
    st.stop()

# Call backend model
probs, backend_classes = call_backend(sig)

# Basic consistency check
if backend_classes and backend_classes != classes:
    st.error("Class order mismatch between backend and app.")
    st.stop()

thr_array = np.array([thresholds[c] for c in classes])
preds     = (probs >= thr_array).astype(int)
pred_labels = [c for c, v in zip(classes, preds) if v] or ["(none)"]

CLASS_LABELS = {
    "NORM": "NORM — Normal",
    "MI":   "MI — Heart Attack (past/recent)",
    "STTC": "STTC — Abnormal ST/T wave",
    "CD":   "CD — Conduction Problem",
    "HYP":  "HYP — Heart Enlargement",
}
rows = []
for c, p, t, v in zip(classes, probs, thr_array, preds):
    rows.append({
        "code": c,
        "name": CLASS_LABELS.get(c, c),
        "prob": float(p),
        "thr": float(t),
        "flagged": bool(v),
    })

# ── Clickable sections using tabs ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "1. Summary (diagnosis)",
    "2. Confidence for each diagnosis",
    "3. Performance in patient groups",
])

# 1. Summary tab
with tab1:
    st.subheader("Diagnosis result")

    # Compute human labels if available
    true_labels = None
    if truth is not None:
        true_labels = [c for c, v in zip(classes, truth) if v == 1] or ["(none)"]

    # Decide colour for model prediction
    if true_labels is None:
        # User-uploaded ECG: no human label
        model_colour = "blue"
        match_note = "This ECG does not have a human-assigned diagnosis in PTB‑XL (e.g. user upload)."
    else:
        if set(pred_labels) == set(true_labels):
            model_colour = "green"
            match_note = "For this ECG, the model's prediction matches the human diagnosis."
        else:
            model_colour = "red"
            match_note = (
                "For this ECG, the model's prediction does not fully match the human diagnosis."
            )

    col1, col2 = st.columns(2)

    # Left: model prediction (coloured)
    with col1:
        st.markdown(
            f"**Model prediction**<br>"
            f"<span style='color:{model_colour}; font-size:1.1rem;'>"
            f"{', '.join(pred_labels)}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The model flags a diagnosis when its confidence for that diagnosis is at or "
            "above the setting you chose in the left panel."
        )

    # Right: human diagnosis (if available)
    with col2:
        if true_labels is not None:
            st.markdown(
                "**Human-assigned diagnosis**<br>"
                f"<span style='font-size:1.1rem;'>{', '.join(true_labels)}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Labels assigned by cardiologists in the PTB‑XL dataset.")
        else:
            st.markdown(
                "**Human-assigned diagnosis**<br><em>Not available for this ECG</em>",
                unsafe_allow_html=True,
            )
            st.caption(
                "This applies when you upload your own ECG without a PTB‑XL label."
            )

    # Short summary sentence
    st.markdown(f"**Summary for this case:** {match_note}")

    # ECG tracing
    st.subheader("ECG tracing — all 12 leads")
    st.markdown(
        "This is the actual ECG the model analysed. "
        "Each panel shows one of the 12 electrical viewpoints (leads) of the heart "
        "over 10 seconds. The horizontal axis is time; the vertical axis is voltage."
    )
    fig, axes = plt.subplots(6, 2, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(sig[:, i], linewidth=0.8)
        ax.set_ylabel(LEADS[i], rotation=0, labelpad=20)
    axes[-1, 0].set_xlabel("Time (hundredths of a second)")
    axes[-1, 1].set_xlabel("Time (hundredths of a second)")
    fig.tight_layout()
    st.pyplot(fig)

# 2. Confidence tab
with tab2:
    st.subheader("How confident is the model for each condition?")
    st.markdown(
        "The chart below shows the model's confidence for each of the five possible diagnoses, "
        "on a scale from **0** (not at all likely) to **1** (very likely). "
        "A condition appears in the **Model prediction** only when its bar reaches or "
        "exceeds the sensitivity slider set for that condition in the sidebar."
    )

    prob_series = pd.Series(
        probs,
        index=[CLASS_LABELS.get(c, c) for c in classes]
    )
    st.bar_chart(prob_series)
    st.caption(
        "The five conditions are: NORM (normal ECG), MI (myocardial infarction / heart attack), "
        "STTC (ST or T-wave abnormality, often linked to ischaemia), "
        "CD (conduction disturbance, e.g. bundle branch block), "
        "HYP (hypertrophy / enlarged heart chamber). "
        "A single ECG can match more than one condition."
    )

# 3. Patient-group performance tab
with tab3:
    st.header("Does the model work equally for all patients?")
    st.markdown(
        "A model can appear accurate overall but still perform worse for certain groups. "
        "This section puts the current ECG into that wider context."
    )

    # Patient-specific summary
    if truth is not None:
        true_labels_pg = [c for c, v in zip(classes, truth) if v == 1] or ["(none)"]
        if set(pred_labels) == set(true_labels_pg):
            st.markdown(
                "For **this patient**, the model's suggestion matches the confirmed diagnosis "
                f"(*Confirmed:* {', '.join(true_labels_pg)} · *Model:* {', '.join(pred_labels)}*).* "
                "The plots below show how often similar predictions are right or wrong across "
                "different patient groups (sex and age) in the test set."
            )
        else:
            st.markdown(
                "For **this patient**, the model's suggestion does **not** fully match the "
                f"confirmed diagnosis (*Confirmed:* {', '.join(true_labels_pg)} · "
                f"*Model:* {', '.join(pred_labels)}*).* "
                "The plots below show how often similar disagreements happen for different "
                "patient groups in the test set."
            )
    else:
        st.markdown(
            "For uploaded ECGs without a confirmed diagnosis, the plots below summarise how "
            "often the model is right or wrong for different patient groups on the held‑out "
            "test set. This gives a sense of how trustworthy its suggestions are overall."
        )

    st.markdown(
        "Ideally, performance would be similar across all groups. "
        "Large gaps suggest the model may need more diverse training data before clinical use."
    )

    # Two plots side by side
    fcols = st.columns(2)
    common_width = 450

    if (FAIRNESS / "auroc_by_sex.png").exists():
        fcols[0].image(
            str(FAIRNESS / "auroc_by_sex.png"),
            caption=(
                "How accurately the model ranks diagnoses, compared between male and female "
                "patients (higher is better; 1.0 would be perfect)."
            ),
            width=common_width,
        )
    if (FAIRNESS / "auroc_by_age.png").exists():
        fcols[1].image(
            str(FAIRNESS / "auroc_by_age.png"),
            caption=(
                "The same accuracy measure broken down by patient age group. "
                "Younger and older patients tend to be harder to classify."
            ),
            width=common_width+70,
        )

    # Subgroup metrics table
    if (FAIRNESS / "subgroup_metrics.csv").exists():
        st.markdown("**Full breakdown by patient group**")
        st.caption(
            "Each row is a patient subgroup from the test set. "
            "Columns show how well the model performed for that group; "
            "higher numbers are better in all columns."
        )
        st.dataframe(pd.read_csv(FAIRNESS / "subgroup_metrics.csv"))

    st.info(
        "🚨 This is a research demo, not a medical device. "
        "These results must not be used for real patient care or clinical decision‑making."
    )