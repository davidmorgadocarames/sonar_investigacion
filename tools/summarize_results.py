"""Compact, versionable summary of one F3 results folder (D-020).

Reads metrics.json (and the permutation, interpretability and DT-005 JSON files next to it) and
writes the numbers the report needs: configuration and val of every model, test scores with
intervals, within-era scores, chance level, permutation test, interpretability and the SHA-256 of
the weights. It only reads results already computed: nothing is trained and test is not scored.

    python tools/summarize_results.py outputs/results docs/results/eval2.json --evaluation 2
"""

import argparse
import json
from pathlib import Path

TOP = 5


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def ci(scores: dict | None) -> dict | None:
    if scores is None:
        return None
    lo, hi = scores["ci95"]["f1_macro"]
    return {
        "f1_macro": round(scores["f1_macro"], 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "chance_f1_macro": (
            round(scores["chance_f1_macro"], 4) if "chance_f1_macro" in scores else None
        ),
    }


def split_summary(split: dict) -> dict:
    return {
        "segment": {
            **ci(split["segment"]),
            "f1_per_class": {
                k: round(v, 4) for k, v in split["segment"]["f1_per_class"].items()
            },
        },
        "recording": ci(split["recording"]),
        "within_era": {
            era: {
                "classes": scores["classes"],
                "segment": ci(scores["segment"]),
                "recording": ci(scores["recording"]),
            }
            for era, scores in split.get("within_era", {}).items()
        },
    }


def model_summary(entry: dict) -> dict:
    config = entry["config"]
    training = entry["training"]
    summary = {
        "weights": entry["weights"],
        "weights_sha256": entry["weights_sha256"],
        "config": {
            k: config[k]
            for k in ("c", "lr", "weight_decay", "seed", "lr_schedule", "torch_threads")
            if k in config
        },
        "cost": entry["cost"],
        "training_seconds": round(training["seconds"], 2),
        "training_device": (entry.get("training_hardware") or {}).get("gpu")
        or training.get("device", "cpu"),
        "val": split_summary(entry["val"]),
        "test": split_summary(entry["test"]),
    }
    if "seeds" in training:
        summary["val_f1_by_seed"] = [
            round(r["best_val_f1_macro"], 4) for r in training["seeds"]
        ]
        summary["val_f1_seed_mean"] = round(training["seed_val_f1_mean"], 4)
        summary["val_f1_seed_std"] = round(training["seed_val_f1_std"], 4)
    if "grouped_cv" in entry:
        cv = entry["grouped_cv"]
        summary["grouped_cv"] = {
            "mean": round(cv["mean"], 4),
            "std": round(cv["std"], 4),
        }
    return summary


def top_bands(entries: list[dict]) -> list[str]:
    best = sorted(entries, key=lambda b: -b["value"])[:TOP]
    return [f"{b['lo_hz']:g}-{b['hi_hz']:g} Hz" for b in best]


def summarize(results_dir: Path, evaluation: int) -> dict:
    metrics = read(results_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(f"{results_dir / 'metrics.json'} not found")
    permutation = read(results_dir / "permutation_test.json")
    interpretability = read(results_dir / "interpretability.json")
    dt005 = read(results_dir / "dt005_duration.json")
    summary = {
        "evaluation": evaluation,
        "source": str(results_dir.as_posix()),
        "code_sha256_at_evaluation": metrics.get("code_sha256"),
        "evaluation_hardware": {
            k: metrics["hardware"].get(k) for k in ("cpu", "device", "torch_threads")
        },
        "models": {name: model_summary(e) for name, e in metrics["models"].items()},
    }
    if permutation:
        summary["permutation_test"] = {
            k: permutation.get(k)
            for k in (
                "c",
                "procedure",
                "n_permutations",
                "observed_f1_macro",
                "null_mean",
                "null_p95",
                "exceed",
                "p_value",
                "seconds",
            )
        }
    if interpretability:
        era = interpretability["era_check"]
        summary["interpretability"] = {
            "split": interpretability["split"],
            "baseline_top_lofar_bands": top_bands(
                interpretability["baseline"]["lofar"]
            ),
            "cnn_top_lofar_occlusion": {
                name: top_bands(e["occlusion_lofar"])
                for name, e in interpretability["cnn"].items()
            },
            "era_lines_hz": era["era_lines_hz"],
            "top_difference_bands_hz": era["top_difference_bands_hz"],
            "era_check_by_model": {
                name: {
                    "spearman_rho": round(m["spearman_rho"], 3),
                    "spearman_p": round(m["spearman_p"], 3),
                    "top_overlap": m["top_overlap"],
                }
                for name, m in era["models"].items()
            },
        }
    if dt005:
        summary["dt005_duration_val"] = {
            "segment": ci(dt005["val"]["segment"]),
            "recording": ci(dt005["val"]["recording"]),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluation", type=int, required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.results_dir, args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
