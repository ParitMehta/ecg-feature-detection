# api/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import threading
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from src.model import ECGNet

# Global state — populated in background thread
_model = None
_device = None
_classes = []
_ready = False
_load_error = None


def _load_model_background():
    global _model, _device, _classes, _ready, _load_error
    try:
        token = os.getenv("HF_TOKEN")
        classes_path = hf_hub_download(
            repo_id="treehugger4/ecg-model",
            filename="processed/class_names.txt",
            repo_type="model",
            token=token
        )
        _classes = open(classes_path).read().splitlines()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = hf_hub_download(
            repo_id="treehugger4/ecg-model",
            filename="best_model.pt",
            repo_type="model",
            token=token
        )
        _model = ECGNet(n_leads=12, n_classes=len(_classes)).to(_device)
        state = torch.load(model_path, map_location=_device)
        _model.load_state_dict(state)
        _model.eval()
        _ready = True
    except Exception as e:
        _load_error = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire model loading in background so port 7860 is free immediately
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()
    yield


app = FastAPI(title="ECGNet API", lifespan=lifespan)


class ECGRequest(BaseModel):
    signal: list[list[float]]


class ECGResponse(BaseModel):
    classes: list[str]
    probabilities: list[float]
    predicted_labels: list[str]


def predict_probs(sig_1000x12: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(sig_1000x12.astype(np.float32)).permute(1, 0).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.sigmoid(_model(x)).cpu().numpy().ravel()
    return probs


@app.get("/")
def healthcheck():
    if _load_error:
        raise HTTPException(status_code=500, detail=f"Model load error: {_load_error}")
    if not _ready:
        return {"status": "loading", "model": "ECGNet"}
    return {"status": "ok", "model": "ECGNet", "n_classes": len(_classes)}


@app.post("/predict", response_model=ECGResponse)
def predict(request: ECGRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please retry shortly")
    arr = np.array(request.signal, dtype=np.float32)
    if arr.shape != (1000, 12):
        return {"classes": _classes, "probabilities": [], "predicted_labels": []}
    probs = predict_probs(arr)
    preds = (probs >= 0.5).astype(int)
    pred_labels = [c for c, v in zip(_classes, preds) if v] or ["(none)"]
    return ECGResponse(
        classes=_classes,
        probabilities=probs.tolist(),
        predicted_labels=pred_labels,
    )
