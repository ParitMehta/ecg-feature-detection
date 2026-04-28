# src/paths.py
from pathlib import Path

ROOT       = Path("/home/mehta/ML Projects/healthtech")
DATASET    = ROOT / "data/physionet.org/files/ptb-xl/1.0.3"
PROCESSED  = ROOT / "data/processed"
REPORTS    = ROOT / "reports"

# Per-stage output folders (auto-created on import)
EDA        = REPORTS / "eda"
PREPROCESS = REPORTS / "preprocess"
TRAIN      = REPORTS / "train"
INSPECT    = REPORTS / "inspect"
FAIRNESS   = REPORTS / "fairness"
MLRUNS     = ROOT / "mlruns"

for p in (PROCESSED, EDA, PREPROCESS, TRAIN, INSPECT, FAIRNESS):
    p.mkdir(parents=True, exist_ok=True)