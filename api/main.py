# api/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from src.model import ECGNet

# Global state — populated during startup
_model = None
_device = None
_classes = []


def load_model_and_classes():
    token = os.getenv("HF_TOKEN")
    classes_path = hf_hub_download(
        repo_id="treehugger4/ecg-model",
        filename="processed/class_names.txt",
        repo_type="model",
        token=token
    )
    classes = open(classes_path).read().splitlines()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(
        repo_id="treehugger4/ecg-model",
        filename="best_model.pt",
        repo_type="model",
        token=token
    )
    model = ECGNet(n_leads=12, n_classes=len(classes)).to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev, classes


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _classes
    _model, _device, _classes = load_model_and_classes()
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
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": "ECGNet", "n_classes": len(_classes)}


@app.post("/predict", response_model=ECGResponse)
def predict(request: ECGRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
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
