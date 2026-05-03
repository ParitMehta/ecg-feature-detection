#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overview page — written for non-technical readers (e.g. medical students)."""

import streamlit as st

st.set_page_config(page_title="Project Overview", layout="wide")
st.title("🫀 ECG Diagnostic Classifier — Project Overview")
st.caption("Summary of the data, model, and evaluation.")

# ── Layout: left = contents, right = main text ───────────────────────────────
left, right = st.columns([1, 3])

with left:
    st.markdown("### Contents")
    section = st.radio(
        "Jump to a topic:",
        [
            "1. Big picture",
            "2. Where ECGs come from",
            "3. Diagnosis groups",
            "4. Train / validation / test",
            "5. File types",
            "6. How the model looks at ECGs",
            "7. Thresholds and sliders",
            "8. How to read the numbers",
            "9. What the demo page shows",
            "10. Limitations",
        ],
        index=0,
    )

with right:
    # 1. Big picture
    if section == "1. Big picture":
        st.header("What is this project about?")
        st.markdown("""
        This project asks a simple question:

        > **Can a computer look at a standard 12-lead ECG and help suggest the right diagnosis?**

        To study this, a machine learning model was trained on thousands of real ECGs.
        It does **not** replace a cardiologist. Instead, it shows:
        - how well a model can learn ECG patterns, and  
        - where it works well or poorly for different patient groups.
        """)

    # 2. Where the data comes from
    if section == "2. Where ECGs come from":
        st.header("Where do the ECGs come from?")
        st.markdown("""
        The ECGs come from **PTB‑XL**, a public ECG database from a German hospital, hosted on
        [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).

        Key facts:

        - Around **21,000 ECG recordings** from real patients  
        - Each recording is **10 seconds** long  
        - Standard **12-lead ECG**, recorded at **100 samples per second**  
        - Each ECG has one or more diagnosis labels assigned by cardiologists  
        - Some ECGs carry **multiple diagnoses** at the same time
        """)

        st.info(
            "Important: PTB‑XL is from a single institution in Germany. The model may not "
            "behave the same way on ECGs from other hospitals or devices."
        )

    # 3. Diagnosis groups
    if section == "3. Diagnosis groups":
        st.header("Which diagnoses does the model predict?")
        st.markdown("The model predicts one or more of these **five** diagnostic groups:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **NORM — Normal ECG**  
            No significant abnormal findings.  
            Most common group in the dataset.

            **MI — Myocardial Infarction**  
            Patterns suggesting a past or ongoing heart attack.  
            Examples: ST elevation, Q waves.

            **STTC — ST/T‑wave Change**  
            Abnormal ST segment or T‑wave (e.g. flattening, inversion, non‑specific changes).  
            Often related to ischaemia or repolarisation problems.
            """)
        with col2:
            st.markdown("""
            **CD — Conduction Disturbance**  
            Problems with how the electrical signal travels through the heart.  
            Examples: bundle branch blocks, AV block patterns.

            **HYP — Hypertrophy**  
            ECG signs that one or more chambers are enlarged.  
            This is the **rarest** class in the data and the **hardest** for the model.
            """)

        st.caption(
            "One ECG can belong to several groups at once, e.g. a patient with both a conduction "
            "disturbance (CD) and hypertrophy (HYP). This is called a *multi‑label* problem."
        )

    # 4. Train / validation / test
    if section == "4. Train / validation / test":
        st.header("How was the data split for training and testing?")
        st.markdown("""
        PTB‑XL provides a built‑in split into 10 folds. This project uses them as follows:

        | Set        | Folds used | What it is for |
        |-----------|-----------|----------------|
        | Training   | 1–8       | The model learns from these ECGs |
        | Validation | 9         | Used to choose good cut‑off values for each diagnosis |
        | Test       | 10        | Kept aside until the end to measure true performance |

        The **demo app only shows results on the test set**, so the numbers you see are for
        ECGs the model did **not** see during training.
        """)

    # 5. File types
    if section == "5. File types":
        st.header("What do the file types mean in this project?")
        st.markdown("""
        You may see several unfamiliar file endings. Here is what they are in plain language:

        | File type | Used for | Think of it as… |
        |---|---|---|
        | `.hea` + `.dat` | Raw PTB‑XL ECGs (WFDB format) | A pair of files: `.hea` is the text header (patient details, sampling rate); `.dat` is the actual signal. |
        | `.npy` | Processed ECG arrays and labels | A compact file storing a big table of numbers (e.g. ECG traces). |
        | `.csv` | Metrics and summaries | A spreadsheet you can open in Excel (e.g. metrics for each subgroup). |
        | `.png` | Plots and figures | Images of ECGs, confusion matrices, or metric charts. |
        | `.pt` | Trained model | The model’s “memory”: weights learned during training. |
        | `thresholds.json` | Decision thresholds | A small text file storing one cut‑off value for each diagnosis. |
        """)

        st.markdown("""
        In the demo:

        - `X_test.npy`, `y_test.npy`, and `class_names.txt` provide the ECGs and diagnoses used.  
        - `best_model.pt` and `thresholds.json` are loaded to make predictions and to set good cut‑offs.
        """)

    # 6. How the model looks at ECGs
    if section == "6. How the model looks at ECGs":
        st.header("How does the model \"look\" at an ECG?")
        st.markdown("""
        The model is a **1D convolutional neural network**. You do **not** need to know the math,
        but it helps to have an intuition:

        1. It reads each ECG as **12 time series** (one per lead) with 1,000 points each.  
        2. It slides small “windows” across the signal, learning to recognise shapes like QRS
           complexes, ST elevations, or T‑wave changes.  
        3. It combines information across time and across leads.  
        4. Finally, it outputs **five probabilities** — one for each diagnosis group.

        The output is always on a **0 to 1 scale** for each diagnosis:
        - 0 = “model sees no evidence for this diagnosis”
        - 1 = “model is very confident this diagnosis is present”
        """)

    # 7. Thresholds and sliders
    if section == "7. Thresholds and sliders":
        st.header("What is a decision threshold? Why does the slider matter?")
        st.markdown("""
        The model always produces **probabilities**, not yes/no answers.  
        To turn a probability into a label, you choose a **decision threshold**.

        Example for myocardial infarction (MI):

        - If the model gives MI a probability of **0.82** and the threshold is **0.80**  
          → the app says **“MI present”**.  
        - If the threshold were **0.90** with the same probability  
          → the app says **“MI not flagged”**.

        In this project:

        - Each diagnosis has its **own threshold**.  
        - Those thresholds were chosen on a separate validation set to balance “catching
          disease” vs “avoiding false alarms” (using a metric called F1 score).

        In the app:

        - The **sidebar sliders** let you move these thresholds up or down for live exploration.  
        - Lowering a slider makes the model more **sensitive** for that diagnosis.  
        - Raising a slider makes it more **cautious**.
        """)

    # 8. How to read the numbers
    if section == "8. How to read the numbers":
        st.header("How should I read the evaluation numbers?")
        st.markdown("""
        The project reports a few common metrics. Here they are in clinical terms:
        """)

        with st.expander("AUROC — \"How well does the model rank sick vs healthy?\"", expanded=True):
            st.markdown("""
            **AUROC** (Area Under the Receiver Operating Characteristic curve):

            - Imagine randomly choosing **one patient with the condition** and **one without**.  
            - AUROC is the chance that the model gives the patient with the condition a **higher**
              probability than the one without.

            Rough guide:
            - 0.5  → as bad as random guessing  
            - 0.7–0.8 → fair  
            - 0.8–0.9 → good  
            - > 0.9 → very good

            This model reaches a **macro AUROC of about 0.90** on the test set — meaning it
            does a good job of ranking patients by risk overall.
            """)

        with st.expander("F1 score — \"Balance between catching disease and avoiding false alarms\""):
            st.markdown("""
            **F1 score** balances two things:

            - **Recall**: “Of all patients who really have this condition, how many did the model catch?”  
            - **Precision**: “Of all patients the model said had this condition, how many really did?”

            F1 is high only when **both** are good. It is useful when conditions are **unequally common**,
            like hypertrophy being much rarer than normal ECGs.

            In this project, F1 is strong for common classes like NORM and MI, and weaker for HYP —
            which has fewer examples and is harder to learn.
            """)

        with st.expander("Fairness metrics — \"Does it work equally well for everyone?\""):
            st.markdown("""
            The project also checks performance separately for:

            - Male vs female patients  
            - Four age groups: <40, 40–60, 60–75, and 75+

            This helps reveal whether the model is **systematically worse** for any subgroup.
            For example, AUROC and F1 are slightly lower for the oldest patients and for women,
            which would need attention before any real clinical use.
            """)

    # 9. What the demo page shows
    if section == "9. What the demo page shows":
        st.header("What does the demo page show?")
        st.markdown("""
        When you open the main **ECG Diagnostic Classifier** page, you will see three main areas:

        1. **Left panel — choosing and tuning**
           - At the top, you choose the **source of the ECG**:
             - *Browse test set (preloaded)*: pick any patient from the held‑out test set.
               These are real PTB‑XL recordings the model did **not** train on.
             - *Upload WFDB record*: upload a `.hea` + `.dat` pair from PTB‑XL.
             - *Upload .npy*: upload a 1000×12 NumPy array with your own ECG.
           - Below that, you see **one slider per diagnosis**. Each slider sets how
             confident the model must be before it flags that condition. Lower = more
             sensitive, higher = more cautious.

        2. **Top of the main area — diagnosis summary**
           - **Model prediction**: which of the five diagnosis groups the model flags, with colour
             indicating whether it matches the human labels (on PTB‑XL cases).  
           - **Human-assigned diagnosis** (when browsing the test set): what the cardiologist
             labelled this ECG as in PTB‑XL.

        3. **Middle and bottom — details and fairness**
           - A **bar chart** showing the model's confidence for each diagnosis on a 0–1 scale.
           - A **12‑lead ECG plot**: the actual tracing the model analysed, one panel per lead.
           - A **patient‑group section** (“Does the model work equally for all patients?”)
             showing how performance changes for different sexes and age bands.
        """)

    # 10. Limitations
    if section == "10. Limitations":
        st.header("Limitations — what this model is NOT")
        st.error("""
        🚨 This is a **research tool**, not a medical device.

        - It was trained and tested on data from **one institution**.  
        - Its performance **differs** across sex and age groups.  
        - The probabilities it outputs are **not calibrated** and should not be used directly
          as risk scores.

        **Do not** use this app to make clinical decisions or to manage real patients.
        It is designed for learning, exploration, and research only.
        """)

# ── Footer: authorship, GitHub, GDPR ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
**About this project**

- Developed by: *Parit Mehta*  
- Source code and documentation: [GitHub repository](https://github.com/ParitMehta/ecg-feature-detection)  
- Contact: *parit.mehta@protonmail.com* for feedback or bug reports  
"""
)

st.markdown(
    """
---
**Data protection notice**

**What data this app processes**  
This app can process two types of input:

- **Preloaded ECGs** from the PTB‑XL research database. These recordings are already fully anonymised by the data providers (Physikalisch-Technische Bundesanstalt, Germany) before they were made publicly available. No names, dates of birth, hospital IDs, or any other identifiers are present in this dataset.  
- **User-uploaded ECG files** (`.hea` / `.dat` or `.npy` format), if you choose to upload one.

**What happens to uploaded files**  
1. Your file is read into the server's working memory (RAM) for the duration of your request.  
2. For WFDB uploads (`.hea` + `.dat`), a short-lived temporary folder is created on the server's disk solely to allow the file to be read. This folder is deleted automatically as soon as the signal has been extracted — typically within a few seconds.  
3. The extracted signal is sent over HTTP to a model API running on the same server, which returns only probabilities. The API does not store the signal.  
4. Once your result is displayed, the signal exists only in the server's RAM for the current session. It is not written to any database, log file, or long-term storage by this application.  
5. Uploaded files are never used to retrain or update the model.

**What this app does not do**  
- It does not collect or store names, contact details, or any information that could identify a person.  
- It does not use cookies or tracking technologies beyond those set by the Streamlit framework itself.  
- It does not share uploaded data with any third party.

**Infrastructure logs**  
This app is hosted on **Streamlit Community Cloud**, operated by Snowflake Inc. Streamlit Community Cloud may retain standard infrastructure logs (e.g. IP address, timestamp, browser type) as part of its normal operation. These logs are outside the control of this application. For details, see the [Streamlit Privacy Policy](https://streamlit.io/privacy-policy).

**Scope and legal basis**  
This app is a **research and education tool** only. It does not constitute a medical device or a clinical service. No special-category health data (within the meaning of GDPR Article 9) is collected or retained by this application. Use of this app is voluntary and no personal data is required to use it.
"""
)