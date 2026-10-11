"""CNN (F3-T07, D-018): two branches, LOFARgram (2D) and DEMON (1D), joined by a small head.

Training: AdamW (no weight decay on biases or BatchNorm), learning rate halved when val macro F1
stops improving for 3 epochs (ReduceLROnPlateau), minibatches of 32 shuffled with a fixed seed,
class-weighted cross-entropy, at most 60 epochs with early stopping (patience 10) on the
segment-level macro F1 of val. Search lr x weight decay with one seed, then train the best
configuration with 3 seeds and keep the best on val.

Reproducibility: seeds fixed, CPU threads fixed, deterministic algorithms required. Shuffling and
augmentation use separate generators, so cnn and cnn-aug see the minibatches in the same order.
The pooling splits the frequency axis into equal groups without padding (no adaptive pooling,
whose CUDA backward is not deterministic). The input normalisation (mean and standard deviation of
each branch on train) is stored in the weights as buffers. (D-018 adjustment 2.)

Test is never loaded here.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sonar_pipeline.dataset import CLASSES, class_weights, load_split
from sonar_pipeline.features import code_sha256
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

LR_GRID = (1e-3, 3e-4)
WEIGHT_DECAY_GRID = (1e-3, 1e-2)
SEEDS = (0, 1, 2)
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10
PLATEAU_FACTOR = 0.5  # lr multiplied by this when val macro F1 stalls
PLATEAU_PATIENCE = 3  # epochs without improvement before lowering the lr
TORCH_THREADS = 4  # fixed CPU threads (D-018 adjustment 2)
AUGMENT_SEED_OFFSET = 10_000  # augmentation generator seed = run seed + offset
DROPOUT = 0.3
LEAKY_SLOPE = 0.01
POOLED_BINS = 8  # frequency positions kept by each branch before the head

# Data augmentation for cnn-aug (F3-T08, D-018): train only, each with probability 0.5.
AUG_PROBABILITY = 0.5
AUG_LOFAR_MASKS = 2  # up to 2 frequency bands masked in the LOFARgram
AUG_LOFAR_MASK_HZ = 100.0  # each at most 100 Hz wide
AUG_TIME_MASK_FRAMES = 2  # one block of up to 2 frames
AUG_SHIFT_FRAMES = 2  # circular time shift of up to +-2 frames
AUG_DEMON_MASK_HZ = 10 / 3  # one DEMON band at most 3.3 Hz wide


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
    """Average the last axis into `bins` contiguous groups, with no padding.

    Group i covers [floor(i * size / bins), ceil((i + 1) * size / bins)), as adaptive average
    pooling does, but built from slices and means, which are deterministic on CUDA.
    """
    size = x.shape[-1]  # with size < bins, groups overlap but none is empty
    groups = [
        x[..., (i * size) // bins : -(-((i + 1) * size) // bins)].mean(dim=-1)
        for i in range(bins)
    ]
    return torch.stack(groups, dim=-1)


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
    torch.set_num_threads(TORCH_THREADS)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    generator = torch.Generator().manual_seed(seed)  # minibatch order
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
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
                    lofar, demon = augment(lofar, demon, augment_generator)
                optimizer.zero_grad()
                loss = loss_fn(model(lofar, demon), y_train[rows])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(rows)
            probs = predict_proba(model, tensors["val_lofar"], tensors["val_demon"])
            val_f1 = present_f1_macro(labels["val"], probs.argmax(axis=1))
            best.history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / n,
                    "val_f1_macro": val_f1,
                    "lr": optimizer.param_groups[0]["lr"],  # lr used in this epoch
                }
            )
            scheduler.step(val_f1)
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


def make_augment(
    lofar_fill: float,
    demon_fill: float,
    lofar_bin_hz: float,
    demon_bin_hz: float,
    probability: float = AUG_PROBABILITY,
):
    """Return augment(lofar, demon, generator) for one training minibatch.

    Masked values are set to the train mean of their branch (0 after normalisation). Every
    random draw comes from `generator`, so a seed gives the same augmentation. The inputs are
    not modified in place.
    """
    lofar_width = max(1, round(AUG_LOFAR_MASK_HZ / lofar_bin_hz))
    demon_width = max(1, round(AUG_DEMON_MASK_HZ / demon_bin_hz))

    def augment(lofar: torch.Tensor, demon: torch.Tensor, generator: torch.Generator):
        def draw(high: int) -> int:
            return int(torch.randint(high, (1,), generator=generator))

        def happens() -> bool:
            return float(torch.rand(1, generator=generator)) < probability

        lofar, demon = lofar.clone(), demon.clone()
        n_frames, n_bins = lofar.shape[1:]
        for i in range(len(lofar)):
            if happens():  # frequency bands of the LOFARgram
                for _ in range(1 + draw(AUG_LOFAR_MASKS)):
                    width = 1 + draw(min(lofar_width, n_bins))
                    start = draw(n_bins - width + 1)
                    lofar[i, :, start : start + width] = lofar_fill
            if happens():  # a block of frames
                width = 1 + draw(min(AUG_TIME_MASK_FRAMES, n_frames))
                start = draw(n_frames - width + 1)
                lofar[i, start : start + width, :] = lofar_fill
            if happens():  # circular time shift of 1 to AUG_SHIFT_FRAMES frames
                shift = 1 + draw(AUG_SHIFT_FRAMES)
                if happens():
                    shift = -shift
                lofar[i] = torch.roll(lofar[i], shifts=shift, dims=0)
            if happens():  # a DEMON band
                width = 1 + draw(min(demon_width, demon.shape[1]))
                start = draw(demon.shape[1] - width + 1)
                demon[i, start : start + width] = demon_fill
        return lofar, demon

    augment.config = {
        "probability": probability,
        "lofar_masks": f"1 to {AUG_LOFAR_MASKS} bands of 1 to {lofar_width} bins "
        f"(up to {AUG_LOFAR_MASK_HZ:g} Hz)",
        "time_mask": f"1 block of 1 to {AUG_TIME_MASK_FRAMES} frames",
        "time_shift": f"circular, 1 to {AUG_SHIFT_FRAMES} frames either way",
        "demon_mask": f"1 band of 1 to {demon_width} bins (up to {AUG_DEMON_MASK_HZ:.1f} Hz)",
        "fill": "train mean of the branch",
    }
    return augment


def train_cnn_aug(
    processed_dir: Path,
    models_dir: Path,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """The CNN of F3-T07 with the same lr, weight decay and seeds, trained with augmentation.

    Reads the configuration from models/cnn.json, so the plain CNN must be trained first.
    """
    import json

    plain = json.loads((models_dir / "cnn.json").read_text(encoding="utf-8"))
    train = load_split(processed_dir, "train")
    augment = make_augment(
        lofar_fill=float(train.lofar.mean()),
        demon_fill=float(train.demon.mean()),
        lofar_bin_hz=float(train.lofar_hz[1] - train.lofar_hz[0]),
        demon_bin_hz=float(train.demon_hz[1] - train.demon_hz[0]),
    )
    report = train_cnn(
        processed_dir,
        models_dir,
        name="cnn_aug",
        device_name=device_name,
        n_bootstrap=n_bootstrap,
        max_epochs=max_epochs,
        patience=patience,
        lr_grid=(plain["config"]["lr"],),
        weight_decay_grid=(plain["config"]["weight_decay"],),
        seeds=seeds,
        augment=augment,
        augment_config=augment.config,
    )
    report["comparison"] = {
        "against": "cnn (same lr, weight decay and seeds, without augmentation), on val",
        "cnn_seed_val_f1": [r["best_val_f1_macro"] for r in plain["training"]["seeds"]],
        "cnn_aug_seed_val_f1": [
            r["best_val_f1_macro"] for r in report["training"]["seeds"]
        ],
        "cnn_seed_val_f1_mean": plain["training"]["seed_val_f1_mean"],
        "cnn_aug_seed_val_f1_mean": report["training"]["seed_val_f1_mean"],
    }
    write_metrics(models_dir / "cnn_aug.json", report)
    return report


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
            "augment_seed_offset": AUGMENT_SEED_OFFSET,
            "lr_schedule": f"ReduceLROnPlateau on val macro F1: x{PLATEAU_FACTOR} after "
            f"{PLATEAU_PATIENCE} epochs without improvement",
            "torch_threads": TORCH_THREADS,
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
        "code_sha256": code_sha256(),
        "traceability": traceability(processed_dir),
        "weights": weights_path.name,
    }
    write_metrics(models_dir / f"{name}.json", report)
    return report
