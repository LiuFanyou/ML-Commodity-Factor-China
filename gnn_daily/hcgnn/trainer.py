from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import HCGNNDaily, info_nce


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def task_loss(
    task: str, pred: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        valid = mask.bool() & torch.isfinite(target)
        if not valid.any():
            return pred.sum() * 0.0
        pred, target = pred[valid], target[valid]
    if task == "cpd":
        return nn.functional.binary_cross_entropy_with_logits(pred, target)
    return nn.functional.mse_loss(pred, target)


class ContinualTrainer:
    def __init__(self, model: HCGNNDaily, cfg: dict[str, Any], device: torch.device):
        self.model, self.cfg, self.device = model.to(device), cfg, device
        self.memories: list[dict[str, dict[str, torch.Tensor]]] = []

    def _regularizer(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        named = dict(self.model.named_parameters())
        for memory in self.memories:
            for name, old in memory["params"].items():
                lam = self.cfg["lambda_heads"] if name.startswith("heads.") else self.cfg["lambda_encoder"]
                reg = reg + lam * (memory["omega"][name] * (named[name] - old).pow(2)).sum()
        return reg

    def _loader(self, data: np.lib.npyio.NpzFile, split: str, task: str, shuffle: bool) -> DataLoader:
        x = torch.from_numpy(data[f"{split}_x"])
        y = torch.from_numpy(data[f"{split}_{task}"])
        mask_key = f"{split}_{task}_mask"
        mask = torch.from_numpy(data[mask_key]) if mask_key in data else torch.ones_like(y, dtype=torch.bool)
        return DataLoader(TensorDataset(x, y, mask), batch_size=self.cfg["batch_size"],
                          shuffle=shuffle, num_workers=self.cfg["num_workers"])

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, task: str) -> float:
        self.model.eval()
        losses, count = 0.0, 0
        for x, y, mask in loader:
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            valid_count = int((mask.bool() & torch.isfinite(y)).sum().item())
            if valid_count == 0:
                continue
            pred, _, _ = self.model(x, task)
            losses += task_loss(task, pred, y, mask).item() * valid_count
            count += valid_count
        return losses / max(count, 1)

    def _consolidate(self, loader: DataLoader, task: str) -> None:
        self.model.eval()
        for x, y, mask in loader:
            if (mask.bool() & torch.isfinite(y)).any():
                break
        else:
            raise ValueError(f"No valid labels available for task {task}")
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.zero_grad(set_to_none=True)
        z1, _ = self.model.encode(x)
        z2, _ = self.model.encode(x + 0.02 * torch.randn_like(x))
        mi_proxy = info_nce(z1, z2)
        pred, _, _ = self.model(x, task)
        # The supervised term gives the active downstream head its own Ω_W,
        # while InfoNCE supplies the heterogeneous-task Ω_Θ signal.
        (mi_proxy + task_loss(task, pred, y, mask)).backward()
        params, omega = {}, {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().clone()
            omega[name] = (param.grad.detach().abs().clone() if param.grad is not None
                           else torch.zeros_like(param))
        self.memories.append({"params": params, "omega": omega})

    def fit(self, data: np.lib.npyio.NpzFile, tasks: list[str]) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for task in tasks:
            train_loader = self._loader(data, "train", task, True)
            val_loader = self._loader(data, "val", task, False)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"],
                                          weight_decay=self.cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
            best, best_state, stale = float("inf"), None, 0
            for epoch in range(self.cfg["epochs_per_task"]):
                self.model.train()
                total, count = 0.0, 0
                for x, y, mask in train_loader:
                    x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                    valid_count = int((mask.bool() & torch.isfinite(y)).sum().item())
                    if valid_count == 0:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    pred, _, _ = self.model(x, task)
                    loss = task_loss(task, pred, y, mask) + self._regularizer()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    total += loss.item() * valid_count
                    count += valid_count
                val = self.evaluate(val_loader, task)
                row = {"task": task, "epoch": epoch + 1, "train_loss": total / max(count, 1),
                       "val_loss": val}
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if val < best:
                    best, best_state, stale = val, copy.deepcopy(self.model.state_dict()), 0
                else:
                    stale += 1
                scheduler.step()
                if stale >= self.cfg["patience"]:
                    break
            if best_state is not None:
                self.model.load_state_dict(best_state)
            self._consolidate(train_loader, task)
        return history


@torch.no_grad()
def predict(model: HCGNNDaily, data: np.lib.npyio.NpzFile, split: str,
            batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(torch.from_numpy(data[f"{split}_x"])), batch_size=batch_size)
    predictions = []
    model.eval()
    for (x,) in loader:
        pred, _, _ = model(x.to(device), "price")
        predictions.append(pred.cpu().numpy())
    scaled = np.concatenate(predictions)
    raw = scaled * data["price_std"] + data["price_mean"]
    return scaled, raw


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape_percent": float(100.0 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-6))),
        "correlation": float(np.corrcoef(pred.ravel(), y.ravel())[0, 1]),
    }


def topk_backtest(y: np.ndarray, pred: np.ndarray, k: int, threshold: float, cost: float) -> dict[str, Any]:
    daily = []
    positions = []
    for actual, signal in zip(y, pred):
        order = np.argsort(signal)
        pos = np.zeros_like(signal)
        longs = [i for i in order[::-1] if signal[i] > threshold][:k]
        shorts = [i for i in order if signal[i] < -threshold][:k]
        if longs:
            pos[longs] = 0.5 / len(longs)
        if shorts:
            pos[shorts] = -0.5 / len(shorts)
        turnover = np.abs(pos - (positions[-1] if positions else 0)).sum()
        daily.append(float(np.dot(pos, actual) - cost * turnover))
        positions.append(pos)
    returns = np.asarray(daily)
    equity = np.cumprod(1.0 + returns)
    ann = np.sqrt(242) * returns.mean() / (returns.std() + 1e-12)
    peak = np.maximum.accumulate(equity)
    return {"total_return": float(equity[-1] - 1), "annualized_sharpe": float(ann),
            "max_drawdown": float(np.min(equity / peak - 1)), "daily_returns": returns.tolist()}
