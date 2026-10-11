"""CNN (F3-T07, D-018): two branches, LOFARgram (2D) and DEMON (1D), joined by a small head.

Training: AdamW (no weight decay on biases or BatchNorm), cosine learning rate, minibatches of 32
shuffled with a fixed seed, class-weighted cross-entropy, at most 60 epochs with early stopping
(patience 10) on the segment-level macro F1 of val. Search lr x weight decay with one seed, then
train the best configuration with 3 seeds and keep the best on val.

Reproducibility: seeds fixed, deterministic algorithms required. The pooling is fixed-size
(no adaptive pooling, whose CUDA backward is not deterministic). The input normalisation (mean
and standard deviation of each branch on train) is stored in the weights as buffers.

Test is never loaded here.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sonar_pipeline.dataset import CLASSES, class_weights, load_split
from sonar_pipeline.metrics import (
    evaluate_split,
    f1_scores,
    hardware,
    stopwatch,
    torch_cost,
    traceability,
    write_metrics,
)

# Deterministic cuBLAS needs this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # after the environment variable
from torch import nn
from torch.nn import functional as F

LR_GRID = (1e-3, 3e-4)
WEIGHT_DECAY_GRID = (1e-3, 1e-2)
SEEDS = (0, 1, 2)
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10
DROPOUT = 0.3
LEAKY_SLOPE = 0.01
POOLED_BINS = 8  # frequency positions kept by each branch before the head


def conv_block_2d(c_in, c_out, kernel, pool):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, padding=(kernel[0] // 2, kernel[1] // 2)),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(LEAKY_SLOPE),
        nn.MaxPool2d((1, pool)),
    )


def conv_block_1d(c_in, c_out, kernel, pool):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.LeakyReLU(LEAKY_SLOPE),
        nn.MaxPool1d(pool),
    )


def pool_to(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Average the last axis into `bins` equal groups (zero-padded at the end if needed).

    A fixed average pooling: deterministic on CUDA, unlike adaptive pooling.
    """
    size = x.shape[-1]
    width = -(-size // bins)
    x = F.pad(x, (0, width * bins - size))
    return F.avg_pool1d(x, width)


class SonarCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.lofar = nn.Sequential(
            conv_block_2d(1, 16, (3, 7), 4),
            conv_block_2d(16, 32, (3, 5), 4),
            conv_block_2d(32, 64, (3, 3), 2),
        )
        self.demon = nn.Sequential(
            conv_block_1d(1, 16, 7, 4),
            conv_block_1d(16, 32, 5, 4),
        )
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(64 * POOLED_BINS + 32 * POOLED_BINS, 64),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(64, n_classes),
        )
        # Input normalisation (scalars per branch), set from train and saved with the weights.
        for name in ("lofar_mean", "demon_mean"):
            self.register_buffer(name, torch.zeros(()))
        for name in ("lofar_std", "demon_std"):
            self.register_buffer(name, torch.ones(()))

    def set_normalisation(self, lofar: np.ndarray, demon: np.ndarray) -> None:
        self.lofar_mean.fill_(float(lofar.mean()))
        self.lofar_std.fill_(float(lofar.std()))
        self.demon_mean.fill_(float(demon.mean()))
        self.demon_std.fill_(float(demon.std()))

    def forward(self, lofar: torch.Tensor, demon: torch.Tensor) -> torch.Tensor:
        """lofar (B, frames, bins), demon (B, bins) -> logits (B, n_classes)."""
        a = ((lofar - self.lofar_mean) / self.lofar_std).unsqueeze(1)
        a = self.lofar(a).mean(dim=2)  # average over time: (B, 64, freq)
        a = pool_to(a, POOLED_BINS).flatten(1)
        b = ((demon - self.demon_mean) / self.demon_std).unsqueeze(1)
        b = pool_to(self.demon(b), POOLED_BINS).flatten(1)
        return self.head(torch.cat([a, b], dim=1))


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available (see compose.gpu.yaml, D-019)"
        )
    return torch.device(name)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def split_parameters(model: nn.Module) -> list[dict]:
    """Weight decay only on conv and linear weights: biases and BatchNorm (1-D) are excluded."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim == 1 else decay).append(parameter)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    lofar: torch.Tensor,
    demon: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(lofar), batch_size):
        logits = model(
            lofar[start : start + batch_size], demon[start : start + batch_size]
        )
        out.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(out).double().numpy()


def present_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    macro, _ = f1_scores(y_true, y_pred, sorted(set(y_true.tolist())))
    return macro


@dataclass
class RunResult:
    lr: float
    weight_decay: float
    seed: int
    best_val_f1: float
    best_epoch: int
    epochs_run: int
    seconds: float
    history: list[dict] = field(default_factory=list)
    state: dict | None = None  # best weights, on CPU


def train_run(
    tensors: dict,
    labels: dict,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    normalisation: tuple[np.ndarray, np.ndarray] | None = None,
    augment=None,
) -> RunResult:
    """One training run with early stopping on val macro F1; keeps the best weights."""
    set_determinism(seed)
    model = SonarCNN()
    model.set_normalisation(*normalisation)
    model.to(device)
    optimizer = torch.optim.AdamW(
        split_parameters(model), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    generator = torch.Generator().manual_seed(seed)
    y_train = torch.as_tensor(labels["train"], device=device)
    n = len(y_train)

    best = RunResult(lr, weight_decay, seed, -1.0, -1, 0, 0.0)
    with stopwatch() as elapsed:
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(n, generator=generator).to(device)
            total = 0.0
            for start in range(0, n, batch_size):
                rows = order[start : start + batch_size]
                lofar, demon = (
                    tensors["train_lofar"][rows],
                    tensors["train_demon"][rows],
                )
                if augment is not None:
                    lofar, demon = augment(lofar, demon, generator)
                optimizer.zero_grad()
                loss = loss_fn(model(lofar, demon), y_train[rows])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(rows)
            scheduler.step()
            probs = predict_proba(model, tensors["val_lofar"], tensors["val_demon"])
            val_f1 = present_f1_macro(labels["val"], probs.argmax(axis=1))
            best.history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / n,
                    "val_f1_macro": val_f1,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            best.epochs_run = epoch
            if val_f1 > best.best_val_f1:
                best.best_val_f1, best.best_epoch = val_f1, epoch
                best.state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            elif epoch - best.best_epoch >= patience:
                break
    best.seconds = elapsed["seconds"]
    return best


def to_tensors(train, val, device: torch.device) -> dict:
    return {
        "train_lofar": torch.as_tensor(train.lofar, device=device),
        "train_demon": torch.as_tensor(train.demon, device=device),
        "val_lofar": torch.as_tensor(val.lofar, device=device),
        "val_demon": torch.as_tensor(val.demon, device=device),
    }


def load_cnn(path: Path, device: torch.device | str = "cpu") -> SonarCNN:
    state = torch.load(path, map_location=device, weights_only=True)
    model = SonarCNN()
    model.load_state_dict(state)
    return model.to(device).eval()


def run_summary(run: RunResult) -> dict:
    return {
        "lr": run.lr,
        "weight_decay": run.weight_decay,
        "seed": run.seed,
        "best_val_f1_macro": run.best_val_f1,
        "best_epoch": run.best_epoch,
        "epochs_run": run.epochs_run,
        "seconds": run.seconds,
    }


def train_cnn(
    processed_dir: Path,
    models_dir: Path,
    name: str = "cnn",
    device_name: str = "auto",
    n_bootstrap: int = 2000,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr_grid: tuple[float, ...] = LR_GRID,
    weight_decay_grid: tuple[float, ...] = WEIGHT_DECAY_GRID,
    seeds: tuple[int, ...] = SEEDS,
    augment=None,
    augment_config: dict | None = None,
) -> dict:
    device = select_device(device_name)
    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")
    tensors = to_tensors(train, val, device)
    labels = {"train": train.labels, "val": val.labels}
    weights = torch.as_tensor(class_weights(train.labels), dtype=torch.float32)
    common = {
        "tensors": tensors,
        "labels": labels,
        "weights": weights,
        "device": device,
        "max_epochs": max_epochs,
        "patience": patience,
        "normalisation": (train.lofar, train.demon),
        "augment": augment,
    }

    with stopwatch() as total_time:
        search = [
            train_run(lr=lr, weight_decay=wd, seed=seeds[0], **common)
            for lr in lr_grid
            for wd in weight_decay_grid
        ]
        chosen = max(search, key=lambda r: r.best_val_f1)  # first on ties
        runs = [chosen] + [
            train_run(lr=chosen.lr, weight_decay=chosen.weight_decay, seed=s, **common)
            for s in seeds[1:]
        ]
    best = max(runs, key=lambda r: r.best_val_f1)

    model = SonarCNN()
    model.load_state_dict(best.state)
    model.to(device).eval()
    with stopwatch() as inference_time:
        probabilities = predict_proba(model, tensors["val_lofar"], tensors["val_demon"])
        if device.type == "cuda":
            torch.cuda.synchronize()

    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"{name}.pt"
    torch.save(best.state, weights_path)
    example = (
        torch.zeros(1, *train.lofar.shape[1:]),
        torch.zeros(1, train.demon.shape[1]),
    )
    cpu_model = SonarCNN()
    cpu_model.load_state_dict(best.state)
    seed_f1 = [r.best_val_f1 for r in runs]
    report = {
        "model": name,
        "description": "two-branch CNN on LOFARgram and DEMON (D-018)",
        "config": {
            "lr": best.lr,
            "weight_decay": best.weight_decay,
            "seed": best.seed,
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": patience,
            "dropout": DROPOUT,
            "lr_grid": list(lr_grid),
            "weight_decay_grid": list(weight_decay_grid),
            "seeds": list(seeds),
            "augment": augment_config,
            "selection": "segment-level macro F1 on val (early stopping and choice)",
        },
        "cost": torch_cost(cpu_model, example),
        "training": {
            "seconds": best.seconds,
            "total_seconds": total_time["seconds"],
            "device": str(device),
            "segments": len(train),
            "search": [run_summary(r) for r in search],
            "seeds": [run_summary(r) for r in runs],
            "seed_val_f1_mean": float(np.mean(seed_f1)),
            "seed_val_f1_std": float(np.std(seed_f1)),
            "history": best.history,
        },
        "inference": {"seconds": inference_time["seconds"], "segments": len(val)},
        "val": evaluate_split(probabilities, val.labels, val.wav_path, n_bootstrap),
        "hardware": hardware(str(device)),
        "traceability": traceability(processed_dir),
        "weights": weights_path.name,
    }
    write_metrics(models_dir / f"{name}.json", report)
    return report
