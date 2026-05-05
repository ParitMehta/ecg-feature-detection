# api/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from src.model import ECGNet

app = FastAPI(title="ECGNet API")

class ECGRequest(BaseModel):
    signal: list[list[float]]

class ECGResponse(BaseModel):
    classes: list[str]
    probabilities: list[float]
    predicted_labels: list[str]

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

model, device, CLASSES = load_model_and_classes()

def predict_probs(sig_1000x12: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(sig_1000x12.astype(np.float32)).permute(1, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy().ravel()
    return probs

@app.get("/")
def healthcheck():
    return {"status": "ok", "model": "ECGNet", "n_classes": len(CLASSES)}

@app.post("/predict", response_model=ECGResponse)
def predict(request: ECGRequest):
    arr = np.array(request.signal, dtype=np.float32)
    if arr.shape != (1000, 12):
        return {"classes": CLASSES, "probabilities": [], "predicted_labels": []}
    probs = predict_probs(arr)
    preds = (probs >= 0.5).astype(int)
    pred_labels = [c for c, v in zip(CLASSES, preds) if v] or ["(none)"]
    return ECGResponse(
        classes=CLASSES,
        probabilities=probs.tolist(),
        predicted_labels=pred_labels,
    )
