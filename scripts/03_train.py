#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL — baseline 1D-CNN with MLflow experiment tracking."""

#%% Imports and project paths
import sys
from pathlib import Path
sys.path.insert(0, "/home/mehta/ML Projects/healthtech")

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch

from src.model import ECGNet

from src.paths import PROCESSED, REPORTS

REPORTS = REPORTS / "train"           # stage-specific subfolder
REPORTS.mkdir(parents=True, exist_ok=True)


def main():
    mlflow.set_tracking_uri(
        "file:///home/mehta/ML Projects/healthtech/mlruns"
    )
    mlflow.set_experiment("ptbxl_baseline_cnn")

    # Load splits
    def load(n): return np.load(PROCESSED / f"{n}.npy")
    X_train, y_train = load("X_train"), load("y_train")
    X_val,   y_val   = load("X_val"),   load("y_val")
    X_test,  y_test  = load("X_test"),  load("y_test")
    class_names = (PROCESSED / "class_names.txt").read_text().splitlines()

    # Replace any residual NaN/Inf (from filtfilt edge effects on quiet leads)
    for arr in (X_train, X_val, X_test):
        np.nan_to_num(arr, copy=False,
                      nan=0.0, posinf=0.0, neginf=0.0)

    def to_tensor(X, y):
        X_t = torch.from_numpy(X).float().permute(0, 2, 1)   # (N, 12, 1000)
        y_t = torch.from_numpy(y).float()
        return X_t, y_t

    X_train_t, y_train_t = to_tensor(X_train, y_train)
    X_val_t,   y_val_t   = to_tensor(X_val,   y_val)
    X_test_t,  y_test_t  = to_tensor(X_test,  y_test)

    params = dict(
        batch_size=128,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        channels="32-64-128",
        sampling_rate=100,
        loss="BCEWithLogitsLoss+pos_weight",
        optimiser="Adam",
        seed=0,
    )
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class-imbalance weights
    pos = y_train_t.sum(dim=0)
    neg = y_train_t.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1)).to(device)
    print("pos_weight:", dict(zip(class_names, pos_weight.cpu().tolist())))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=params["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=params["batch_size"],
                              num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                              batch_size=params["batch_size"],
                              num_workers=0)

    def train_one_epoch(model, loader, loss_fn, optim):
        model.train(); total = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item() * X.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def evaluate(model, loader, loss_fn):
        model.eval(); total = 0.0
        ys, ps = [], []
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total += loss_fn(logits, y).item() * X.size(0)
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
        y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
        aurocs = []
        for k in range(y_true.shape[1]):
            if 0 < y_true[:, k].sum() < len(y_true):
                aurocs.append(roc_auc_score(y_true[:, k], y_prob[:, k]))
            else:
                aurocs.append(float("nan"))
        macro = float(np.nanmean(aurocs))
        return total / len(loader.dataset), macro, aurocs

    with mlflow.start_run(run_name="baseline-cnn") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train_t))
        mlflow.log_param("n_val",   len(X_val_t))
        mlflow.log_param("n_test",  len(X_test_t))
        mlflow.log_dict({"class_names": class_names}, "class_names.json")

        model   = ECGNet().to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim   = torch.optim.Adam(model.parameters(),
                                   lr=params["learning_rate"],
                                   weight_decay=params["weight_decay"])

        best_val_auc = -1.0
        for epoch in range(1, params["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optim)
            val_loss, val_auc, _ = evaluate(model, val_loader, loss_fn)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_macro_auroc": val_auc,
            }, step=epoch)

            print(f"epoch {epoch:02d} "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"val_macro_auroc={val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), REPORTS / "best_model.pt")

        # ---- Test on the best checkpoint ----
        model.load_state_dict(torch.load(REPORTS / "best_model.pt"))
        test_loss, test_auc, test_aurocs = evaluate(model, test_loader, loss_fn)
        per_class = dict(zip(class_names, test_aurocs))

        mlflow.log_metric("test_loss",        test_loss)
        mlflow.log_metric("test_macro_auroc", test_auc)
        for cls, a in per_class.items():
            if not np.isnan(a):
                mlflow.log_metric(f"test_auroc_{cls}", a)

        print(f"\ntest_loss={test_loss:.4f}  test_macro_auroc={test_auc:.4f}")
        print("per-class test AUROC:", per_class)

        # Small human-readable report
        report = (
            f"best val macro-AUROC: {best_val_auc:.4f}\n"
            f"test macro-AUROC:     {test_auc:.4f}\n"
            f"per-class test AUROC: {per_class}\n"
        )
        (REPORTS / "report.txt").write_text(report)
        mlflow.log_artifact(str(REPORTS / "report.txt"))

        # Log the model itself
        sample = X_val_t[:1].numpy()
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=sample,
        )
        print("MLflow run id:", run.info.run_id)


if __name__ == "__main__":
    main()