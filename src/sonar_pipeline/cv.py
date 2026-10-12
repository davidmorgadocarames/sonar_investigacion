"""Round 3 (F5, D-024): the 7 models under 5 x 5 repeated cross-validation by vessel.

For every model:

1. Selection: every configuration is trained with seed 0 on the 25 partitions (train, early
   stopping on val). The configuration with the best mean val macro F1 (4 classes, per segment)
   wins; ties go to the first one, the simplest. Test is not loaded in this stage.
2. Final: the chosen configuration with seeds 0, 1 and 2 on every partition (seed 0 is the
   selection run, which is bit-for-bit the same training, so it is reused); the test
   probabilities of the three seeds are averaged and scored.

Every training writes its weights under ``models/round3/<model>/`` and a JSON under
``outputs/round3/<model>/`` (MLOps traceability, cost, speed, learning curve, results, partition
data); a training whose JSON and weights exist with the same configuration and code is not
repeated, so the round can be resumed. Afterwards the pre-registered comparisons use the
Nadeau and Bengio corrected t-test with Holm's correction, and ``docs/results/round3.json``
collects the summaries. The models run one after another (blocks of D-024); if one fails, its
error is written and the others go on.
"""

import csv
import hashlib
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sonar_pipeline import baseline as linear
from sonar_pipeline.cnn import (
    AUGMENT_SEED_OFFSET,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    PLATEAU_FACTOR,
    PLATEAU_PATIENCE,
    TORCH_THREADS,
    SonarCNN,
    make_augment,
    predict_proba,
    select_device,
    set_determinism,
)
from sonar_pipeline.dataset import (
    CLASSES,
    aggregate_by_recording,
    baseline_features,
    class_weights,
    load_split,
)
from sonar_pipeline.features import SPLITS, code_sha256
from sonar_pipeline.metrics import (
    evaluate_split,
    f1_scores,
    hardware,
    torch_cost,
    traceability,
    write_metrics,
)
from sonar_pipeline.transformer import SonarHybrid, SonarTransformer

MODELS = (
    "baseline",
    "cnn",
    "cnn_aug",
    "hybrid",
    "hybrid_aug",
    "transformer",
    "transformer_aug",
)
# Order of execution (D-024): baseline and CNN, hybrid, transformer.
BLOCKS = (
    ("baseline", "cnn", "cnn_aug"),
    ("hybrid", "hybrid_aug"),
    ("transformer", "transformer_aug"),
)
SEEDS = (0, 1, 2)
SELECTION_SEED = 0
WARMUP_EPOCHS = 5  # transformer and hybrid: linear warmup of the lr...
WARMUP_START = 0.1  # ...from 10 % of it
PARTITION_BOOTSTRAP = 200  # bootstrap resamples of each partition's intervals
CALIBRATION_BINS = 10
LATENCY_REPEATS = 20
SEGMENT_SECONDS = 3.0

_TRANSFORMER = {
    "A": {"dim": 32, "layers": 1, "heads": 2, "groups": 1, "dropout": 0.3},
    "B": {"dim": 64, "layers": 1, "heads": 4, "groups": 2, "dropout": 0.3},
    "C": {"dim": 64, "layers": 2, "heads": 4, "groups": 2, "dropout": 0.3},
    "D": {"dim": 64, "layers": 2, "heads": 4, "groups": 2, "dropout": 0.5},
}
_CNN_A = {
    "lofar_channels": [16, 32, 64],
    "demon_channels": [16, 32],
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-2,
}
# Configurations of D-024, the simplest first (ties go to it).
CONFIGS = {
    "baseline": {
        name: {"c": c} for name, c in zip("ABCDE", linear.C_GRID, strict=True)
    },
    "cnn": {
        "A": _CNN_A,
        "B": {**_CNN_A, "lofar_channels": [8, 16, 32], "demon_channels": [8, 16]},
        "C": {**_CNN_A, "dropout": 0.5, "weight_decay": 5e-2},
        "D": {**_CNN_A, "lr": 3e-4},
    },
    "transformer": {
        name: {**c, "lr": 5e-4, "weight_decay": 5e-2}
        for name, c in _TRANSFORMER.items()
    },
    "hybrid": {
        name: {**c, "lr": 5e-4, "weight_decay": 5e-2}
        for name, c in _TRANSFORMER.items()
    },
}
DESCRIPTIONS = {
    "baseline": "logistic regression on the time-mean LOFAR and the DEMON (D-018)",
    "cnn": "two-branch CNN on LOFARgram and DEMON (D-018)",
    "transformer": "transformer on 40 Hz LOFAR bands, DEMON and class tokens: RMSNorm, RoPE, "
    "GQA, SwiGLU (D-024)",
    "hybrid": "two CNN blocks on the LOFARgram as tokens, DEMON token from the 1D CNN branch, "
    "transformer blocks as the transformer (D-024)",
}

# Pre-registered comparisons (D-024): (model, against).
COMPARISONS = (
    ("cnn", "baseline"),
    ("cnn_aug", "baseline"),
    ("hybrid", "baseline"),
    ("hybrid_aug", "baseline"),
    ("transformer", "baseline"),
    ("transformer_aug", "baseline"),
    ("cnn_aug", "cnn"),
    ("hybrid_aug", "hybrid"),
    ("transformer_aug", "transformer"),
    ("transformer", "cnn"),
    ("hybrid", "cnn"),
    ("transformer_aug", "cnn_aug"),
    ("hybrid_aug", "cnn_aug"),
)


def family(model: str) -> str:
    return model.removesuffix("_aug")


def augmented(model: str) -> bool:
    return model.endswith("_aug")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- data


@dataclass
class Pool:
    """Every segment of the three F2 splits together; partitions are made from cv_splits.csv."""

    segment_id: np.ndarray
    lofar: np.ndarray
    demon: np.ndarray
    labels: np.ndarray
    wav_path: np.ndarray
    vessel_group: np.ndarray
    lofar_hz: np.ndarray
    demon_hz: np.ndarray

    def __len__(self) -> int:
        return len(self.segment_id)


def load_pool(processed_dir: Path) -> Pool:
    parts = [load_split(processed_dir, split) for split in SPLITS]
    order = np.argsort(np.concatenate([p.segment_id for p in parts]), kind="stable")

    def joined(name):
        return np.concatenate([getattr(p, name) for p in parts])[order]

    return Pool(
        segment_id=joined("segment_id"),
        lofar=joined("lofar"),
        demon=joined("demon"),
        labels=joined("labels"),
        wav_path=joined("wav_path"),
        vessel_group=joined("vessel_group"),
        lofar_hz=parts[0].lofar_hz,
        demon_hz=parts[0].demon_hz,
    )


@dataclass(frozen=True)
class Partition:
    repetition: int
    fold: int
    roles: dict = field(hash=False)  # wav_path -> train / val / test

    @property
    def name(self) -> str:
        return f"rep{self.repetition}_fold{self.fold}"


def read_partitions(path: Path) -> list[Partition]:
    roles: dict[tuple[int, int], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["repetition"]), int(row["fold"]))
            roles.setdefault(key, {})[row["wav_path"]] = row["split"]
    return [Partition(r, k, roles[(r, k)]) for r, k in sorted(roles)]


def partition_rows(pool: Pool, partition: Partition) -> dict[str, np.ndarray]:
    """Rows of every role; checks coverage, no vessel in two roles, 4 classes in each role."""
    recordings = set(pool.wav_path.tolist())
    if recordings != set(partition.roles):
        raise ValueError(
            f"{partition.name}: the partition and the features cover different recordings"
        )
    role = np.array([partition.roles[w] for w in pool.wav_path])
    rows = {r: np.flatnonzero(role == r) for r in ("train", "val", "test")}
    seen: dict[str, str] = {}
    for r, idx in rows.items():
        for group in set(pool.vessel_group[idx].tolist()):
            if seen.setdefault(group, r) != r:
                raise ValueError(
                    f"{partition.name}: vessel {group} in {seen[group]} and {r}"
                )
        missing = set(range(len(CLASSES))) - set(pool.labels[idx].tolist())
        if missing:
            names = [CLASSES[k] for k in sorted(missing)]
            raise ValueError(f"{partition.name}: {r} has no {names}")
    return rows


def partition_data(pool: Pool, rows: dict[str, np.ndarray]) -> dict:
    """Segments, recordings, vessels and minutes of every role, per class."""
    summary = {}
    for role, idx in rows.items():
        per_class = {}
        for k, name in enumerate(CLASSES):
            mine = idx[pool.labels[idx] == k]
            per_class[name] = {
                "segments": len(mine),
                "recordings": len(set(pool.wav_path[mine].tolist())),
                "vessels": sorted(set(pool.vessel_group[mine].tolist())),
                "minutes": len(mine) * SEGMENT_SECONDS / 60,
            }
        summary[role] = {
            "segments": len(idx),
            "recordings": len(set(pool.wav_path[idx].tolist())),
            "vessels": len(set(pool.vessel_group[idx].tolist())),
            "minutes": len(idx) * SEGMENT_SECONDS / 60,
            "per_class": per_class,
        }
    return summary


# --------------------------------------------------------------------------- networks


def build_network(model: str, config: dict, pool_shapes: tuple) -> nn.Module:
    frames, lofar_bins, demon_bins = pool_shapes
    fam = family(model)
    if fam == "cnn":
        return SonarCNN(
            lofar_channels=tuple(config["lofar_channels"]),
            demon_channels=tuple(config["demon_channels"]),
            dropout=config["dropout"],
        )
    cls = {"transformer": SonarTransformer, "hybrid": SonarHybrid}[fam]
    return cls(
        dim=config["dim"],
        layers=config["layers"],
        heads=config["heads"],
        groups=config["groups"],
        dropout=config["dropout"],
        lofar_frames=frames,
        lofar_bins=lofar_bins,
        demon_bins=demon_bins,
    )


def split_parameters(model: nn.Module) -> list[dict]:
    """Weight decay only on weight matrices and filters: biases, BatchNorm, RMSNorm gains and
    the class token are excluded."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        skip = parameter.ndim == 1 or name == "cls_token"
        (no_decay if skip else decay).append(parameter)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def uses_warmup(model: str) -> bool:
    return family(model) in ("transformer", "hybrid")


def warmup_lr(base: float, epoch: int) -> float:
    """lr of a warmup epoch (1-based): linear from WARMUP_START * base, base after warmup."""
    if epoch > WARMUP_EPOCHS:
        return base
    return base * (WARMUP_START + (1 - WARMUP_START) * (epoch - 1) / WARMUP_EPOCHS)


@dataclass
class NetRun:
    history: list = field(default_factory=list)
    best_val_f1: float = -1.0
    best_epoch: int = -1
    epochs_run: int = 0
    seconds: float = 0.0
    segments_seen: int = 0
    gpu_peak_bytes: int | None = None
    gpu_reserved_bytes: int | None = None
    state: dict | None = None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def val_scores(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(macro F1 of the 4 classes, mean cross-entropy) on val."""
    f1, _ = f1_scores(labels, probs.argmax(axis=1))
    p_true = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return f1, float(-np.log(p_true).mean())


def fit_network(
    model_name: str,
    config: dict,
    data: dict,
    seed: int,
    device: torch.device,
    augment=None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[nn.Module, NetRun]:
    """Train on data["train_*"] with early stopping on data["val_*"]; returns the best model.

    data: train_lofar, train_demon, train_labels, val_lofar, val_demon, val_labels (tensors on
    the device, labels as numpy) and normalisation (train lofar and demon arrays). No test here.
    """
    set_determinism(seed)
    shapes = tuple(data["train_lofar"].shape[1:]) + (data["train_demon"].shape[1],)
    model = build_network(model_name, config, shapes)
    model.set_normalisation(*data["normalisation"])
    model.to(device)
    lr = config["lr"]
    warmup = uses_warmup(model_name)
    optimizer = torch.optim.AdamW(
        split_parameters(model),
        lr=warmup_lr(lr, 1) if warmup else lr,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
    )
    weights = torch.as_tensor(class_weights(data["train_labels"]), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    generator = torch.Generator().manual_seed(seed)
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
    y_train = torch.as_tensor(data["train_labels"], device=device)
    n = len(y_train)
    run = NetRun()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        if warmup and epoch <= WARMUP_EPOCHS:
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr(lr, epoch)
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for first in range(0, n, BATCH_SIZE):
            rows = order[first : first + BATCH_SIZE]
            lofar, demon = data["train_lofar"][rows], data["train_demon"][rows]
            if augment is not None:
                lofar, demon = augment(lofar, demon, augment_generator)
            optimizer.zero_grad()
            loss = loss_fn(model(lofar, demon), y_train[rows])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        train_seconds = time.perf_counter() - epoch_start
        probs = predict_proba(model, data["val_lofar"], data["val_demon"])
        val_f1, val_loss = val_scores(probs, data["val_labels"])
        synchronize(device)
        run.history.append(
            {
                "epoch": epoch,
                "train_loss": total / n,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
        run.segments_seen += n
        run.epochs_run = epoch
        if not warmup or epoch >= WARMUP_EPOCHS:  # the plateau rule starts after warmup
            scheduler.step(val_f1)
        if val_f1 > run.best_val_f1:
            run.best_val_f1, run.best_epoch = val_f1, epoch
            run.state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        elif epoch - run.best_epoch >= patience:
            break
    run.seconds = time.perf_counter() - start
    if device.type == "cuda":
        run.gpu_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        run.gpu_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    model.load_state_dict(run.state)
    return model.eval(), run


def latency(model: nn.Module, lofar, demon, device) -> dict:
    """Seconds of one forward pass on a single segment (median of LATENCY_REPEATS) and the
    throughput of predict_proba on all the given segments."""
    one = (lofar[:1], demon[:1])
    times = []
    with torch.no_grad():
        model(*one)  # warm-up
        for _ in range(LATENCY_REPEATS):
            synchronize(device)
            t = time.perf_counter()
            model(*one)
            synchronize(device)
            times.append(time.perf_counter() - t)
    synchronize(device)
    t = time.perf_counter()
    predict_proba(model, lofar, demon)
    synchronize(device)
    seconds = time.perf_counter() - t
    return {
        "single_segment_seconds_median": float(np.median(times)),
        "batch_segments": len(lofar),
        "batch_seconds": seconds,
        "segments_per_second": len(lofar) / seconds if seconds > 0 else None,
    }


# --------------------------------------------------------------------------- scoring


def calibration(
    probs: np.ndarray, labels: np.ndarray, bins: int = CALIBRATION_BINS
) -> dict:
    """Expected calibration error of the top-class probability, with its reliability bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
        ece += mask.mean() * abs(acc - conf)
        table.append(
            {
                "lo": edges[b],
                "hi": edges[b + 1],
                "n": int(mask.sum()),
                "accuracy": acc,
                "confidence": conf,
            }
        )
    return {"ece": float(ece), "bins": table}


def recording_predictions(probs, labels, wav_path) -> list[dict]:
    rec = aggregate_by_recording(probs, wav_path, labels)
    return [
        {
            "wav_path": str(w),
            "label": CLASSES[int(y)],
            "predicted": CLASSES[int(p.argmax())],
            "probabilities": dict(zip(CLASSES, p.tolist(), strict=True)),
        }
        for w, y, p in zip(rec.wav_path, rec.labels, rec.probabilities, strict=True)
    ]


def score_block(probs, labels, wav_path, n_bootstrap: int) -> dict:
    """Full scores of one role (as in metrics.json), calibration and per-recording output."""
    return {
        **evaluate_split(probs, labels, wav_path, n_bootstrap),
        "calibration": calibration(probs, labels),
        "recordings": recording_predictions(probs, labels, wav_path),
    }


def nadeau_bengio(a, b, test_train_ratio: float) -> dict:
    """Corrected resampled t-test (Nadeau and Bengio, 2003) of the per-partition scores a - b.

    Variance inflated by (1 / J + n_test / n_train) for the overlap between training sets.
    """
    from scipy import stats

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = len(d)
    mean, var = float(d.mean()), float(d.var(ddof=1)) if j > 1 else 0.0
    scale = (1 / j + test_train_ratio) * var
    if scale <= 0:
        t, p = (0.0, 1.0) if mean == 0 else (float("inf") * np.sign(mean), 0.0)
    else:
        t = mean / np.sqrt(scale)
        p = float(2 * stats.t.sf(abs(t), df=j - 1))
    return {
        "mean_difference": mean,
        "std_difference": float(np.sqrt(var)),
        "partitions": j,
        "test_train_ratio": test_train_ratio,
        "t": float(t),
        "df": j - 1,
        "p": p,
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


# --------------------------------------------------------------------------- engine


class Round3:
    """Trains, resumes and summarises the round. Paths are injected so tests use tmp folders."""

    def __init__(
        self,
        processed_dir: Path,
        splits_path: Path,
        models_dir: Path,
        outputs_dir: Path,
        device_name: str = "auto",
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        n_bootstrap: int = PARTITION_BOOTSTRAP,
        configs: dict | None = None,
        seeds: tuple[int, ...] = SEEDS,
        log=print,
    ):
        self.processed_dir = processed_dir
        self.splits_path = splits_path
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.device = select_device(device_name)
        self.max_epochs, self.patience = max_epochs, patience
        self.n_bootstrap = n_bootstrap
        self.configs = configs or CONFIGS
        self.seeds = seeds
        self.log = log
        self.pool = load_pool(processed_dir)
        self.partitions = read_partitions(splits_path)
        self.rows = {p.name: partition_rows(self.pool, p) for p in self.partitions}
        self.code = code_sha256()
        self.trace = traceability(processed_dir)
        self.splits_sha256 = file_sha256(splits_path)
        self.hardware = hardware(str(self.device))
        self._device_pool = None
        self._cost_cache: dict = {}

    # ---- helpers

    def device_pool(self) -> dict:
        if self._device_pool is None:
            self._device_pool = {
                "lofar": torch.as_tensor(self.pool.lofar, device=self.device),
                "demon": torch.as_tensor(self.pool.demon, device=self.device),
            }
        return self._device_pool

    def net_data(self, partition: Partition) -> dict:
        rows, dp = self.rows[partition.name], self.device_pool()
        data = {
            "normalisation": (
                self.pool.lofar[rows["train"]],
                self.pool.demon[rows["train"]],
            )
        }
        for role in ("train", "val"):  # never test
            idx = torch.as_tensor(rows[role], device=self.device)
            data[f"{role}_lofar"] = dp["lofar"][idx]
            data[f"{role}_demon"] = dp["demon"][idx]
            data[f"{role}_labels"] = self.pool.labels[rows[role]]
        return data

    def test_tensors(self, partition: Partition):
        idx = torch.as_tensor(self.rows[partition.name]["test"], device=self.device)
        dp = self.device_pool()
        return dp["lofar"][idx], dp["demon"][idx]

    def augment_for(self, model: str, partition: Partition):
        if not augmented(model):
            return None
        train = self.rows[partition.name]["train"]
        return make_augment(
            lofar_fill=float(self.pool.lofar[train].mean()),
            demon_fill=float(self.pool.demon[train].mean()),
            lofar_bin_hz=float(self.pool.lofar_hz[1] - self.pool.lofar_hz[0]),
            demon_bin_hz=float(self.pool.demon_hz[1] - self.pool.demon_hz[0]),
        )

    def cost(self, model: str, config_name: str) -> dict:
        key = (family(model), config_name)
        if key not in self._cost_cache:
            config = self.configs[family(model)][config_name]
            if family(model) == "baseline":
                from sonar_pipeline.metrics import linear_model_cost

                n_features = self.pool.lofar.shape[2] + self.pool.demon.shape[1]
                cost = {**linear_model_cost(n_features), "tokens_per_segment": None}
            else:
                shapes = self.pool.lofar.shape[1:] + (self.pool.demon.shape[1],)
                net = build_network(model, config, shapes)
                example = (
                    torch.zeros(1, *self.pool.lofar.shape[1:]),
                    torch.zeros(1, self.pool.demon.shape[1]),
                )
                cost = {
                    **torch_cost(net, example),
                    "tokens_per_segment": getattr(net, "tokens_per_segment", None),
                }
            cost["mflops_per_segment"] = cost["flops_per_segment"] / 1e6
            self._cost_cache[key] = cost
        return self._cost_cache[key]

    def paths(self, model, stage, config_name, partition, seed):
        if stage == "selection":
            rel = Path(model, "selection", config_name, partition.name)
        else:
            rel = Path(model, "final", partition.name, f"seed{seed}")
        suffix = ".npz" if family(model) == "baseline" else ".pt"
        return (self.models_dir / rel).with_suffix(suffix), (
            self.outputs_dir / rel
        ).with_suffix(".json")

    def fresh(self, record_path: Path, weights_path: Path, config: dict) -> dict | None:
        """The saved record if the training is already done with this config and code."""
        if not (record_path.is_file() and weights_path.is_file()):
            return None
        record = json.loads(record_path.read_text(encoding="utf-8"))
        same = (
            record.get("complete")
            and record["config"]["hyperparameters"] == json.loads(json.dumps(config))
            and record["mlops"]["code_sha256"] == self.code
            and record["mlops"]["cv_splits_sha256"] == self.splits_sha256
            and record["config"]["max_epochs"] == self.max_epochs
            and record["config"]["patience"] == self.patience
        )
        return record if same else None

    # ---- one training

    def train_one(self, model, stage, config_name, partition, seed) -> dict:
        config = self.configs[family(model)][config_name]
        weights_path, record_path = self.paths(
            model, stage, config_name, partition, seed
        )
        done = self.fresh(record_path, weights_path, config)
        if done is not None:
            return done
        started = now()
        if family(model) == "baseline":
            result = self._train_baseline(config, partition, weights_path)
        else:
            result = self._train_network(
                model, config, config_name, partition, seed, weights_path
            )
        record = {
            "complete": True,
            "model": model,
            "description": DESCRIPTIONS[family(model)],
            "stage": stage,
            "partition": {
                "repetition": partition.repetition,
                "fold": partition.fold,
                "data": partition_data(self.pool, self.rows[partition.name]),
            },
            "config": {
                "name": config_name,
                "hyperparameters": config,
                "seed": seed,
                "augment": augmented(model),
                "augment_seed_offset": AUGMENT_SEED_OFFSET,
                "batch_size": BATCH_SIZE,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "optimizer": "AdamW (no weight decay on biases, norms and class token)",
                "lr_schedule": (
                    f"linear warmup {WARMUP_EPOCHS} epochs from {WARMUP_START:g} x lr, then "
                    if uses_warmup(model)
                    else ""
                )
                + f"ReduceLROnPlateau on val macro F1 x{PLATEAU_FACTOR} after "
                f"{PLATEAU_PATIENCE} epochs",
                "loss": "cross-entropy with balanced class weights of train",
                "selection_metric": "segment-level macro F1 of the 4 classes on val",
            },
            "mlops": {
                "started": started,
                "finished": now(),
                "code_sha256": self.code,
                "cv_splits_sha256": self.splits_sha256,
                "traceability": self.trace,
                "hardware": self.hardware,
                "torch_threads": TORCH_THREADS,
                "deterministic": True,
                "weights_path": weights_path.as_posix(),
                "weights_sha256": file_sha256(weights_path),
            },
            "cost": self.cost(model, config_name),
            **result,
        }
        write_metrics(record_path, record)
        return json.loads(record_path.read_text(encoding="utf-8"))

    def _train_baseline(self, config, partition, weights_path) -> dict:
        rows = self.rows[partition.name]
        x = {r: baseline_features(_view(self.pool, rows[r])) for r in ("train", "val")}
        y = {r: self.pool.labels[rows[r]] for r in ("train", "val")}
        start = time.perf_counter()
        fitted = linear.make_model(config["c"]).fit(x["train"], y["train"])
        seconds = time.perf_counter() - start
        linear.save_baseline(fitted, weights_path, config["c"])
        probs = fitted.predict_proba(x["val"])
        f1, loss = val_scores(probs, y["val"])
        return {
            "training": {
                "seconds": seconds,
                "device": "cpu",
                "segments": len(y["train"]),
                "segments_per_second": len(y["train"]) / seconds,
                "note": "deterministic: one fit, the seed does not change it",
            },
            "result": {
                "best_val_f1_macro": f1,
                "val_loss": loss,
                "val": score_block(
                    probs, y["val"], self.pool.wav_path[rows["val"]], self.n_bootstrap
                ),
            },
        }

    def _train_network(self, model, config, config_name, partition, seed, weights_path):
        data = self.net_data(partition)
        net, run = fit_network(
            model,
            config,
            data,
            seed,
            self.device,
            self.augment_for(model, partition),
            self.max_epochs,
            self.patience,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run.state, weights_path)
        probs = predict_proba(net, data["val_lofar"], data["val_demon"])
        cost = self.cost(model, config_name)
        tokens = cost["tokens_per_segment"]
        train_seconds = sum(h["train_seconds"] for h in run.history)
        speed = run.segments_seen / train_seconds if train_seconds > 0 else None
        val_rows = self.rows[partition.name]["val"]
        return {
            "training": {
                "seconds": run.seconds,
                "device": str(self.device),
                "segments": len(data["train_labels"]),
                "segments_seen": run.segments_seen,
                "epochs_run": run.epochs_run,
                "best_epoch": run.best_epoch,
                "seconds_per_epoch_mean": run.seconds / run.epochs_run,
                "segments_per_second": speed,
                "tokens_per_second": speed * tokens if (speed and tokens) else None,
                "training_flops_estimate": 3
                * cost["flops_per_segment"]
                * run.segments_seen,
                "gpu_peak_allocated_bytes": run.gpu_peak_bytes,
                "gpu_peak_reserved_bytes": run.gpu_reserved_bytes,
                "history": run.history,
            },
            "inference": latency(
                net, data["val_lofar"], data["val_demon"], self.device
            ),
            "result": {
                "best_val_f1_macro": run.best_val_f1,
                "val": score_block(
                    probs,
                    data["val_labels"],
                    self.pool.wav_path[val_rows],
                    self.n_bootstrap,
                ),
            },
        }

    # ---- test, only in the final stage

    def predict_test(self, model, config_name, partition, weights_path) -> np.ndarray:
        rows = self.rows[partition.name]["test"]
        if family(model) == "baseline":
            weights = linear.load_baseline(weights_path)
            return linear.predict_proba(
                weights, baseline_features(_view(self.pool, rows))
            )
        shapes = self.pool.lofar.shape[1:] + (self.pool.demon.shape[1],)
        net = build_network(model, self.configs[family(model)][config_name], shapes)
        net.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        net.to(self.device).eval()
        return predict_proba(net, *self.test_tensors(partition))

    # ---- one model

    def run_model(self, model: str) -> dict:
        fam = family(model)
        configs = self.configs[fam]
        model_dir = self.outputs_dir / model
        started = now()
        selection = {}
        for config_name in configs:
            scores = []
            for partition in self.partitions:
                record = self.train_one(
                    model, "selection", config_name, partition, SELECTION_SEED
                )
                scores.append(record["result"]["best_val_f1_macro"])
                self.log(
                    f"[{model}] selection {config_name} {partition.name}: "
                    f"val F1 {scores[-1]:.3f} ({_epochs(record)})"
                )
            selection[config_name] = {
                "val_f1_macro_mean": float(np.mean(scores)),
                "val_f1_macro_std": float(np.std(scores, ddof=1))
                if len(scores) > 1
                else 0.0,
                "per_partition": scores,
            }
        chosen = None
        for config_name, entry in selection.items():  # first on ties
            if (
                chosen is None
                or entry["val_f1_macro_mean"] > selection[chosen]["val_f1_macro_mean"]
            ):
                chosen = config_name
        self.log(
            f"[{model}] chosen configuration {chosen} "
            f"(mean val F1 {selection[chosen]['val_f1_macro_mean']:.3f})"
        )

        seeds = (SELECTION_SEED,) if fam == "baseline" else self.seeds
        partitions = []
        for partition in self.partitions:
            seed_records, seed_probs = [], []
            for seed in seeds:
                if seed == SELECTION_SEED:
                    record = self.train_one(model, "selection", chosen, partition, seed)
                    src, _ = self.paths(model, "selection", chosen, partition, seed)
                    dst, _ = self.paths(model, "final", chosen, partition, seed)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                else:
                    record = self.train_one(model, "final", chosen, partition, seed)
                    dst, _ = self.paths(model, "final", chosen, partition, seed)
                seed_records.append(record)
                seed_probs.append(self.predict_test(model, chosen, partition, dst))
            partitions.append(
                self._score_partition(
                    model, chosen, partition, seeds, seed_records, seed_probs
                )
            )
            self.log(
                f"[{model}] final {partition.name}: test F1 "
                f"{partitions[-1]['test_f1_macro_segment']:.3f} (seeds {list(seeds)})"
            )
        summary = self._summarise(model, chosen, selection, partitions, started)
        write_metrics(model_dir / "summary.json", summary)
        self._copy_representative(model, chosen)
        return summary

    def _score_partition(self, model, chosen, partition, seeds, records, probs) -> dict:
        rows = self.rows[partition.name]["test"]
        labels, wav = self.pool.labels[rows], self.pool.wav_path[rows]
        ensemble = np.mean(probs, axis=0)
        test = score_block(ensemble, labels, wav, self.n_bootstrap)
        per_seed = [f1_scores(labels, p.argmax(axis=1))[0] for p in probs]
        n_train = len(self.rows[partition.name]["train"])
        entry = {
            "model": model,
            "config": chosen,
            "repetition": partition.repetition,
            "fold": partition.fold,
            "seeds": list(seeds),
            "test_f1_macro_segment": test["segment"]["f1_macro"],
            "test_f1_macro_recording": test["recording"]["f1_macro"],
            "test_f1_macro_per_seed": per_seed,
            "test_within_era_segment": {
                era: (v["segment"] or {}).get("f1_macro")
                for era, v in test["within_era"].items()
            },
            "test_chance_f1_macro": test["segment"]["chance_f1_macro"],
            "test_ece": test["calibration"]["ece"],
            "val_f1_macro_per_seed": [
                r["result"]["best_val_f1_macro"] for r in records
            ],
            "n_train_segments": n_train,
            "n_test_segments": len(rows),
            "training_seconds": float(sum(r["training"]["seconds"] for r in records)),
            "gpu_peak_allocated_bytes": max(
                (r["training"].get("gpu_peak_allocated_bytes") or 0) for r in records
            )
            or None,
        }
        detail = {
            **entry,
            "test": test,
            "test_segment_ids": self.pool.segment_id[rows],
            "test_probabilities": ensemble,
            "weights_sha256": [r["mlops"]["weights_sha256"] for r in records],
            "code_sha256": self.code,
            "cv_splits_sha256": self.splits_sha256,
        }
        write_metrics(
            self.outputs_dir / model / "final" / partition.name / "ensemble.json",
            detail,
        )
        entry["_confusion"] = test["segment"]["confusion"]
        entry["_probs"] = (ensemble, labels)
        return entry

    def _summarise(self, model, chosen, selection, partitions, started) -> dict:
        confusion = np.sum([p.pop("_confusion") for p in partitions], axis=0)
        probs = np.concatenate([p["_probs"][0] for p in partitions])
        labels = np.concatenate([p.pop("_probs")[1] for p in partitions])
        totals = confusion.sum(axis=1, keepdims=True)

        def stats(key):
            values = [p[key] for p in partitions]
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        eras = {}
        for era in partitions[0]["test_within_era_segment"]:
            values = [p["test_within_era_segment"][era] for p in partitions]
            values = [v for v in values if v is not None]
            eras[era] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        records = self._all_records(model, chosen)
        gpu = [r["training"].get("gpu_peak_allocated_bytes") for r in records]
        gpu = [g for g in gpu if g]
        return {
            "model": model,
            "description": DESCRIPTIONS[family(model)],
            "augment": augmented(model),
            "started": started,
            "finished": now(),
            "selection": selection,
            "chosen_config": chosen,
            "chosen_hyperparameters": self.configs[family(model)][chosen],
            "cost": self.cost(model, chosen),
            "test": {
                "f1_macro_segment": stats("test_f1_macro_segment"),
                "f1_macro_recording": stats("test_f1_macro_recording"),
                "within_era_segment": eras,
                "chance_f1_macro": stats("test_chance_f1_macro"),
                "ece": stats("test_ece"),
                "joint_confusion": confusion.tolist(),
                "joint_confusion_normalized": np.divide(
                    confusion, totals, out=np.zeros(confusion.shape), where=totals > 0
                ).tolist(),
                "joint_calibration": calibration(probs, labels),
            },
            "partitions": partitions,
            "timing": {
                "trainings": len(records),
                "training_seconds_total": float(
                    sum(r["training"]["seconds"] for r in records)
                ),
                "seconds_per_epoch_mean": float(
                    np.mean([r["training"]["seconds_per_epoch_mean"] for r in records])
                )
                if family(model) != "baseline"
                else None,
                "segments_per_second_mean": float(
                    np.mean([r["training"]["segments_per_second"] for r in records])
                ),
                "tokens_per_second_mean": float(
                    np.mean([r["training"]["tokens_per_second"] for r in records])
                )
                if records and records[0]["training"].get("tokens_per_second")
                else None,
                "gpu_peak_allocated_bytes_max": max(gpu) if gpu else None,
            },
            "code_sha256": self.code,
            "cv_splits_sha256": self.splits_sha256,
        }

    def _all_records(self, model, chosen) -> list[dict]:
        out = []
        for path in sorted((self.outputs_dir / model).rglob("*.json")):
            if path.name in ("summary.json", "ensemble.json"):
                continue
            out.append(json.loads(path.read_text(encoding="utf-8")))
        return out

    def _copy_representative(self, model, chosen) -> None:
        """One versioned model per network: repetition 0, fold 0, seed 0 (D-024 adj. 2)."""
        first = self.partitions[0]
        src, _ = self.paths(model, "final", chosen, first, SELECTION_SEED)
        folder = self.models_dir / "representative"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, folder / f"{model}{src.suffix}")
        write_metrics(
            folder / f"{model}.json",
            {
                "model": model,
                "config": chosen,
                "hyperparameters": self.configs[family(model)][chosen],
                "partition": first.name,
                "seed": SELECTION_SEED,
                "weights_sha256": file_sha256(src),
                "code_sha256": self.code,
                "cv_splits_sha256": self.splits_sha256,
            },
        )

    # ---- the round

    def run(self, models=MODELS, summary_path: Path | None = None) -> dict:
        results, failures = {}, {}
        started = now()
        for block in BLOCKS:
            for model in block:
                if model not in models:
                    continue
                try:
                    results[model] = self.run_model(model)
                # David (2026-10-12): if one model fails, go on with the others.
                except Exception:  # noqa: BLE001
                    text = traceback.format_exc()
                    path = self.outputs_dir / model / "error.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                    failures[model] = text.strip().splitlines()[-1]
                    self.log(f"[{model}] FAILED: {failures[model]} (see {path})")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        # Models finished in earlier runs (other blocks) with the same code and partitions.
        for model in MODELS:
            path = self.outputs_dir / model / "summary.json"
            if model in results or model in failures or not path.is_file():
                continue
            earlier = json.loads(path.read_text(encoding="utf-8"))
            if (earlier["code_sha256"], earlier["cv_splits_sha256"]) == (
                self.code,
                self.splits_sha256,
            ):
                results[model] = earlier
        results = {m: results[m] for m in MODELS if m in results}
        summary = self.round_summary(results, failures, started)
        if summary_path is not None:
            write_metrics(summary_path, summary)
        return summary

    def round_summary(self, results: dict, failures: dict, started: str) -> dict:
        ratio = float(
            np.mean([len(r["test"]) / len(r["train"]) for r in self.rows.values()])
        )
        comparisons = []
        for a, b in COMPARISONS:
            if a in results and b in results:
                fa = [p["test_f1_macro_segment"] for p in results[a]["partitions"]]
                fb = [p["test_f1_macro_segment"] for p in results[b]["partitions"]]
                comparisons.append(
                    {"model": a, "against": b, **nadeau_bengio(fa, fb, ratio)}
                )
        for entry, p_holm in zip(
            comparisons, holm([c["p"] for c in comparisons]), strict=True
        ):
            entry["p_holm"] = p_holm
        compact = {}
        for name, r in results.items():
            compact[name] = {k: v for k, v in r.items() if k not in ("partitions",)}
            compact[name]["per_partition"] = [
                {
                    k: p[k]
                    for k in (
                        "repetition",
                        "fold",
                        "test_f1_macro_segment",
                        "test_f1_macro_recording",
                        "test_f1_macro_per_seed",
                        "test_within_era_segment",
                        "test_chance_f1_macro",
                        "test_ece",
                        "val_f1_macro_per_seed",
                        "training_seconds",
                        "gpu_peak_allocated_bytes",
                    )
                }
                for p in r["partitions"]
            ]
            compact[name]["test"] = {
                k: v for k, v in r["test"].items() if k != "joint_calibration"
            } | {"joint_ece": r["test"]["joint_calibration"]["ece"]}
        return {
            "round": 3,
            "protocol": "D-024 (5 x 5 repeated cross-validation by vessel)",
            "started": started,
            "finished": now(),
            "partitions": len(self.partitions),
            "seeds": list(self.seeds),
            "test_train_ratio": ratio,
            "comparison_metric": "segment-level test macro F1 (4 classes) per partition",
            "comparison_test": "Nadeau and Bengio corrected resampled t-test, Holm correction",
            "models": compact,
            "comparisons": comparisons,
            "failures": failures,
            "code_sha256": self.code,
            "cv_splits_sha256": self.splits_sha256,
            "traceability": self.trace,
            "hardware": self.hardware,
        }

    # ---- measurement before the round

    def measure(self, epochs: int = 2) -> dict:
        """Epoch time and GPU memory of every network configuration on the first partition.

        No early stopping, no files except the returned dict; test is not used.
        """
        partition = self.partitions[0]
        out = {}
        for model in MODELS:
            if family(model) == "baseline":
                continue
            for config_name, config in self.configs[family(model)].items():
                data = self.net_data(partition)
                _, run = fit_network(
                    model,
                    config,
                    data,
                    SELECTION_SEED,
                    self.device,
                    self.augment_for(model, partition),
                    epochs,
                    epochs + 1,
                )
                out[f"{model}/{config_name}"] = {
                    "seconds_per_epoch": run.seconds / run.epochs_run,
                    "gpu_peak_allocated_bytes": run.gpu_peak_bytes,
                    "gpu_peak_reserved_bytes": run.gpu_reserved_bytes,
                    "parameters": self.cost(model, config_name)["parameters"],
                }
                self.log(
                    f"[measure] {model} {config_name}: "
                    f"{out[f'{model}/{config_name}']['seconds_per_epoch']:.2f} s/epoch, "
                    f"GPU peak {_mib(run.gpu_peak_bytes)}"
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return out


def _view(pool: Pool, rows: np.ndarray):
    """A SplitData-like view for baseline_features."""

    @dataclass
    class View:
        lofar: np.ndarray
        demon: np.ndarray

    return View(pool.lofar[rows], pool.demon[rows])


def _epochs(record: dict) -> str:
    t = record["training"]
    if "epochs_run" in t:
        return f"{t['epochs_run']} epochs, best {t['best_epoch']}, {t['seconds']:.1f} s"
    return f"{t['seconds']:.1f} s"


def _mib(value) -> str:
    return "n/a (CPU)" if value is None else f"{value / 2**20:.0f} MiB"
