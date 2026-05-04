# src/paths.py
import os
from pathlib import Path

ROOT      = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATASET   = ROOT / "data/physionet.org/files/ptb-xl/1.0.3"
PROCESSED = ROOT / "data/processed"
REPORTS   = ROOT / "reports"

EDA        = REPORTS / "eda"
PREPROCESS = REPORTS / "preprocess"
TRAIN      = REPORTS / "train"
INSPECT    = REPORTS / "inspect"
FAIRNESS   = REPORTS / "fairness"
MLRUNS     = ROOT / "mlruns"

for p in (PROCESSED, EDA, PREPROCESS, TRAIN, INSPECT, FAIRNESS):
    p.mkdir(parents=True, exist_ok=True)
