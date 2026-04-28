# PTB-XL Project Handover

This document explains the current state of the PTB-XL ECG classification
project. It covers what the data looks like, what the existing code does,
and the known pitfalls to avoid when extending it.

Read it in order. Later sections assume the earlier ones.

## 1. What the project is

The goal is to build a model that takes a 12-lead ECG recording and
predicts which of 5 heart-condition categories apply to it:

- **NORM** — normal
- **MI** — heart attack (past or current)
- **STTC** — signs that part of the heart isn't getting enough oxygen
- **CD** — electrical signal not flowing through the heart correctly
- **HYP** — a chamber of the heart is enlarged

A single recording can have more than one category at the same time. In
machine-learning terms, this is **multi-label classification**: each
input can have several correct labels, not just one.

The dataset used is **PTB-XL v1.0.3**, published on PhysioNet. It
contains 21,799 ten-second ECG recordings from roughly 18,000 patients
at a German hospital, at two sampling rates (100 Hz and 500 Hz), with
doctor-assigned labels.

## 2. How the data is organized on disk

After download, the dataset lives under one folder. In this project:
/home/mehta/ML Projects/healthtech/data/physionet.org/files/ptb-xl/1.0.3/
├── ptbxl_database.csv # one row per recording: metadata and labels
├── scp_statements.csv # a lookup table: meaning of each diagnostic code
├── records100/ # 100 Hz waveform files (faster to load)
└── records500/ # 500 Hz waveform files (more detailed)



Within `records100/` and `records500/`, recordings are split into
subfolders of 1000 records each (`00000/`, `01000/`, …, `21000/`).
Each recording consists of two files:

- `<id>_lr.hea` — text header describing the signal
- `<id>_lr.dat` — binary file holding the actual voltage values

(`_lr` = low resolution / 100 Hz. The 500 Hz versions end in `_hr`.)

### `ptbxl_database.csv`

21,799 rows, 27 columns. One row per recording. The columns used by the
current pipeline are:

| Column | Purpose |
|---|---|
| `ecg_id` (index) | Unique integer identifier for the recording (1–21799). |
| `scp_codes` | A string that looks like a Python dict, e.g. `"{'NORM': 100.0, 'SR': 0.0}"`. Keys are SCP codes (short cardiology abbreviations). Values are the doctor's confidence on a 0–100 scale. This is the raw source of labels. |
| `strat_fold` | Integer 1–10. The pre-computed stratified split fold for that recording. |
| `filename_lr` | Path to the 100 Hz waveform, relative to the dataset root, without extension (e.g. `records100/00000/00001_lr`). |
| `filename_hr` | Same, for the 500 Hz version. |

Other columns exist (patient age, sex, device, noise flags, free-text
report, etc.). They are not used by the current pipeline but may be
useful later.

### `scp_statements.csv`

71 rows, 12 columns. This is a dictionary: each row describes one SCP
code and explains what category it belongs to. The columns used by the
current pipeline are:

| Column | Purpose |
|---|---|
| Index | The SCP code itself (e.g. `NORM`, `IMI`, `LVOLT`). |
| `diagnostic` | `1.0` if the code represents a diagnosis. Blank otherwise (for example, rhythm or morphology codes). |
| `diagnostic_class` | Which of the five big categories the code rolls up into: NORM, MI, STTC, CD, or HYP. |

The remaining columns map SCP codes to other medical coding systems
(DICOM, CDISC, AHA). Not used here.

### How the two CSVs connect

A recording's 5-category labels are derived from its `scp_codes`:

1. Take each key in the recording's `scp_codes` dict.
2. Look it up in `scp_statements.csv`.
3. Keep only those codes where `diagnostic == 1`.
4. Collect their `diagnostic_class` values.
5. Remove duplicates.

Example: a recording with `scp_codes = {'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}`
resolves to the label set `{NORM, CD}`, because `NORM` maps to
diagnostic_class `NORM`, `LVOLT` maps to `CD`, and `SR` is not a
diagnostic code.

## 3. Source code overview

There are currently two code files relevant for the eda.


src/
└── loader.py # reusable functions for loading data
scripts/
└── 01_eda.py # an analysis script that uses loader.py


### 3.1. `src/loader.py`

Four functions. Each does one thing.

#### `load_metadata(path) -> DataFrame`

Reads `ptbxl_database.csv` into a pandas DataFrame with `ecg_id` as the
index. Parses the `scp_codes` column, which is stored as text, into
actual Python dictionaries using `ast.literal_eval`.

- Returns the loaded DataFrame.
- Does not add any labels, does not filter any rows, and does not load
  any waveform files.
- Is fast (~1 second). Safe to call any time.

#### `add_diagnostic_labels(Y, path) -> DataFrame`

Takes a DataFrame (from `load_metadata`) and the dataset path. Reads
`scp_statements.csv`, filters to diagnostic codes, and adds a new
column `diagnostic_superclass` containing the list of categories for
each recording.

- Returns a new DataFrame with the extra column. The input DataFrame is
  not modified.
- Some recordings will end up with an empty list `[]` (see section 4).
- Is fast (~1 second).

#### `load_raw_signals(Y, path, sampling_rate=100) -> ndarray`

Reads the WFDB waveform files listed in `Y` and returns a numpy array
of shape `(number_of_records, time_samples, 12_leads)`.

- `sampling_rate=100` loads the 100 Hz version (1000 samples per record);
  `sampling_rate=500` loads the 500 Hz version (5000 samples per record).
- Output values are in millivolts.
- This function is slow. Loading the full dataset at 100 Hz takes
  approximately 1–2 minutes; at 500 Hz, approximately 5–10 minutes and
  around 2 GB of RAM.
- Do not call this during quick metadata exploration — pass only the
  rows you actually need.

#### `get_splits(X, Y, val_fold=9, test_fold=10) -> tuple`

Splits the data into training, validation, and test sets using the
`strat_fold` column.

- Training set: folds 1–8.
- Validation set: fold 9.
- Test set: fold 10.
- Returns six objects in this order: `X_train, y_train, X_val, y_val,
  X_test, y_test`.
- `y_*` values are pandas Series of `list[str]` (the multi-label
  categories). They still need to be converted to a numeric format
  before being passed to a model.

### 3.2. `scripts/01_eda.py`

A Spyder cell-based analysis script (uses `#%%` cell separators). It
imports functions from `src.loader`, runs a series of checks on the
data, and saves outputs to `reports/eda/`.

At the top:

```python
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")
```

This hardcoded absolute path lets the script find the `src/` package.
Update it if the project is moved.

Then:

```python
PATH = "/home/mehta/ML Projects/healthtech/data/physionet.org/files/ptb-xl/1.0.3"
OUT  = Path("/home/mehta/ML Projects/healthtech/reports/eda")
```

`PATH` points at the dataset. `OUT` is where all generated images,
CSVs, and the text summary are written.

The cells below those, in order:

1. **Load metadata** — calls `load_metadata` and `add_diagnostic_labels`.
2. **Filter to existing files** — keeps only rows whose waveform file
   exists on disk. Acts as a safety net if the download is incomplete.
3. **Label distribution (Q1)** — counts per category, saves bar chart.
4. **Records with no label (Q3)** — counts recordings whose label list
   is empty.
5. **Load a few signals** — loads 5 recordings as a sanity plot.
6. **Multi-label prevalence (Q2)** — how many recordings have 0, 1, 2,
   3, or 4 categories.
7. **Per-fold distribution (Q4)** — class proportions for each of the
   10 folds, saved as CSV + stacked bar chart.
8. **One record per category (Q5)** — visually compares a typical
   NORM, MI, STTC, CD, HYP waveform on lead I.
9. **All 12 leads of one record (Q5b)** — shows what the model's full
   input actually looks like.
10. **Sanity checks (Q6)** — checks for NaNs, voltage range, and dead
    leads on 100 random records.
11. **Write summary** — writes `summary.txt` with all numeric results.

All results from the most recent run are summarized in
`reports/eda/findings.md`, with the raw numbers in
`reports/eda/summary.txt`.

## 4. Current state of the data

Based on the most recent EDA run on the full dataset:

- All 21,799 records are present and loadable.
- Records with no diagnostic superclass: 411 (~1.9%). Working set after
  dropping these: 21,388.
- Multi-label prevalence: about 24% of recordings have 2 or more
  categories.
- Class counts (categories add to more than total because of
  multi-label):
  - NORM: 9,514
  - MI:   5,469
  - STTC: 5,235
  - CD:   4,898
  - HYP:  2,649
- Fold balance: class proportions vary by less than 2.5 percentage
  points between any two folds.
- Data quality: no NaNs, voltage range within ±5 mV, no dead leads in
  the sampled records.

See `reports/eda/findings.md` for the full write-up and decisions
derived from these numbers.

## 5. Pitfalls to avoid

Each of these produces code that runs without error but gives wrong
results. They tend not to be noticed until test-time metrics look off,
by which point it may be unclear what caused the problem.

### 5.1. Do not use softmax or `CrossEntropyLoss`

Approximately 24% of recordings have two or more categories. A
single-label setup (softmax + cross-entropy) forces the model to pick
exactly one, discarding the other labels silently. Use sigmoid output
and `BCEWithLogitsLoss`.

### 5.2. Drop recordings with empty label lists before training

The 411 recordings with `diagnostic_superclass == []` have no learnable
target. Keeping them in training pushes the model toward predicting
"nothing," and the loss value will still look reasonable. Filter them
out before creating training batches.

### 5.3. Do not resplit the data randomly

The `strat_fold` column provides a patient-stratified split: the same
patient never appears in more than one fold. Many patients have
multiple recordings. A random shuffle would place the same patient in
both training and test, producing optimistic but invalid scores. Use
`strat_fold` as it is.

### 5.4. Check the axis order before feeding signals to a model

`load_raw_signals` returns arrays shaped `(records, time, leads)`. Most
1D convolutional architectures in PyTorch expect `(records, leads, time)`
— channels first. Transpose the last two axes once, at the point where
data enters the model. Feeding the wrong order will train without
errors but produce poor results.

### 5.5. Normalize per recording, not across the dataset

Compute mean and standard deviation separately for each recording (or
per lead within each recording) when normalizing. Computing one mean
and standard deviation from the entire dataset requires careful
handling to avoid leaking statistics from the test set into training
data. Per-recording normalization avoids the issue entirely.

### 5.6. Keep 100 Hz and 500 Hz separate

The two sampling rates produce arrays of different sizes (1000 vs 5000
samples). A model trained at one rate cannot be evaluated at the other
without resampling. Always record which `sampling_rate` a given
experiment used.

### 5.7. Do not use `scp_codes` values as labels

The numeric values in `scp_codes` (0–100) are doctor confidence
scores, not probabilities or binary labels. The pipeline's labels come
only from the `diagnostic_class` column in `scp_statements.csv`, via
`add_diagnostic_labels`.

### 5.8. The path references are absolute

`scripts/01_eda.py` contains three hardcoded absolute paths (the
project root for `sys.path`, `PATH` for the dataset, and `OUT` for
outputs). They are tied to the original machine. On a new machine,
these three values must be updated, or the script can be refactored to
read them from environment variables or a configuration file.

## 6. What is not yet built

The following modules are planned but not implemented at the time of
this handover.

- `src/preprocess.py` — per-recording z-score normalization, drop
  recordings with empty labels, convert `diagnostic_superclass` from a
  list of strings to a `(N, 5)` binary indicator array, transpose
  signals to `(N, 12, time)`.
- `src/models.py` — a 1D-CNN baseline model. Final layer must output 5
  values with no activation inside the model.
- `src/training.py` — training loop using `BCEWithLogitsLoss` with
  per-class `pos_weight` values (see `findings.md`). Validation runs on
  fold 9. Model checkpoints saved under `artifacts/`.
- `src/evaluate.py` — computes per-class and macro-averaged AUROC on
  the test fold. Intended to be run once, at the end of the project.

None of these modules should require changes to `src/loader.py`. If a
change to `loader.py` appears necessary while building them, it is
worth re-checking whether a pitfall from section 5 is being introduced.

## 7. Running the existing code

The environment is a conda environment named `ecg` with Python 3.14.
Required packages:

```bash
conda install numpy pandas matplotlib pyyaml
pip install wfdb
```

To reproduce the EDA:

1. Activate the environment: `conda activate ecg`.
2. Open `scripts/01_eda.py` in Spyder.
3. Update the three hardcoded paths at the top if the project lives
   elsewhere.
4. Run each cell in order, or run the whole script.
5. Outputs appear in `reports/eda/`.

## 8. Further reading

- Dataset description and paper:
  https://physionet.org/content/ptb-xl/1.0.3/
- Original example script by the dataset authors:
  https://physionet.org/content/ptb-xl/1.0.3/example_physionet.py
- Benchmark results for comparison:
  Strodthoff et al., "Deep Learning for ECG Analysis: Benchmarks and
  Insights from PTB-XL."
- EDA findings for this project: `reports/eda/findings.md`

