# api/main.py
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.paths import PROCESSED, REPORTS
from src.model import ECGNet

TRAIN_OUT = REPORTS / "train"

app = FastAPI(title="ECGNet API")

# ---------- Request / response schemas ----------
class ECGRequest(BaseModel):
    # 1000x12 array flattened as list of lists
    signal: list[list[float]]  # shape (1000, 12)

class ECGResponse(BaseModel):
    classes: list[str]
    probabilities: list[float]
    predicted_labels: list[str]


# ---------- Load model and metadata once ----------
def load_model_and_classes():
    classes = (PROCESSED / "class_names.txt").read_text().splitlines()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(n_leads=12, n_classes=len(classes)).to(dev)
    state = torch.load(TRAIN_OUT / "best_model.pt", map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev, classes

model, device, CLASSES = load_model_and_classes()


def predict_probs(sig_1000x12: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(sig_1000x12.astype(np.float32)).permute(1, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy().ravel()
    return probs


# ---------- Endpoints ----------
@app.get("/")
def healthcheck():
    return {"status": "ok", "model": "ECGNet", "n_classes": len(CLASSES)}

@app.post("/predict", response_model=ECGResponse)
def predict(request: ECGRequest):
    arr = np.array(request.signal, dtype=np.float32)
    if arr.shape != (1000, 12):
        return {"classes": CLASSES, "probabilities": [], "predicted_labels": []}

    probs = predict_probs(arr)
    # Simple default threshold 0.5 here; your Streamlit app can apply tuned thresholds client-side
    preds = (probs >= 0.5).astype(int)
    pred_labels = [c for c, v in zip(CLASSES, preds) if v] or ["(none)"]

    return ECGResponse(
        classes=CLASSES,
        probabilities=probs.tolist(),
        predicted_labels=pred_labels,
    )
