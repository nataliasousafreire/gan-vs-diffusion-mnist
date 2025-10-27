import os, json
from typing import Dict, Callable, Any, Optional, Sequence
import torch

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _is_better(curr: float, best: float, mode: str, min_delta: float) -> bool:
    if curr != curr:  # NaN
        return False
    if best == float("inf") and mode == "min":
        return True
    if best == -float("inf") and mode == "max":
        return True
    if mode == "min":
        return (best - curr) > min_delta
    elif mode == "max":
        return (curr - best) > min_delta
    else:
        raise ValueError("mode must be 'min' or 'max'")

class EarlyStopper:

    def __init__(self, mode: str = "min", patience: int = 3, min_delta: float = 0.0):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf") if mode == "min" else -float("inf")
        self.wait = 0
        self.stopped_epoch = -1

    def step(self, value: float, epoch: int) -> bool:
        improved = _is_better(value, self.best, self.mode, self.min_delta)
        if improved:
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

class ReduceLROnFID:
    def __init__(self, optimizer, factor=0.5, patience=3, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_fid = float('inf')
        self.num_bad_epochs = 0

    def step(self, current_fid):
        improved = current_fid < self.best_fid
        if improved:
            self.best_fid = current_fid
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f"[ReduceLROnFID] FID não melhorou por {self.patience} epochs "
                              f"reduzindo LR de {old_lr:.2e} para {new_lr:.2e}")            
            self.num_bad_epochs = 0