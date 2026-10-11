"""Does your evaluation reproduce the reference one? (D-023)

Compares outputs/results/metrics.json (written by `evaluate`) with docs/results/eval2.json: the
weights must be the same file (SHA-256) and every test F1 (per segment, per recording, interval
bounds and within each era) must match within a tolerance. On x86-64 the match is exact; on other
processors, ARM above all, the last digit may change, which the tolerance accepts.

    docker compose run --rm --entrypoint python pipeline tools/check_reproduction.py
"""

import argparse
import json
import sys
from pathlib import Path

TOLERANCE = 0.005


def values(test: dict) -> dict[str, float]:
    """Flat {name: value} of the test F1 figures of one model (full or summary format)."""

    def f1(scores):
        lo, hi = (
            scores["ci95"]["f1_macro"]
            if isinstance(scores["ci95"], dict)
            else scores["ci95"]
        )
        return scores["f1_macro"], lo, hi

    flat = {}
    for level in ("segment", "recording"):
        flat[f"{level}"], flat[f"{level} ci low"], flat[f"{level} ci high"] = f1(
            test[level]
        )
    for era, scores in test["within_era"].items():
        if scores["segment"] is not None:
            flat[f"{era} segment"] = scores["segment"]["f1_macro"]
    return flat


def compare(metrics: dict, reference: dict, tolerance: float = TOLERANCE) -> list[str]:
    """Problems found; an empty list means the evaluation reproduces the reference."""
    problems = []
    for name, ref in reference["models"].items():
        mine = metrics["models"].get(name)
        if mine is None:
            problems.append(f"{name}: missing in metrics.json")
            continue
        if mine["weights_sha256"] != ref["weights_sha256"]:
            problems.append(
                f"{name}: different weights (did you retrain? git checkout -- models/)"
            )
        expected, got = values(ref["test"]), values(mine["test"])
        for key, value in expected.items():
            if abs(got[key] - value) > tolerance:
                problems.append(f"{name} {key}: {got[key]:.4f} instead of {value:.4f}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrics", type=Path, default=Path("outputs/results/metrics.json")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("docs/results/eval2.json")
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)
    if not args.metrics.is_file():
        print(
            f"{args.metrics} not found: run `evaluate` first (README).", file=sys.stderr
        )
        return 2
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    problems = compare(metrics, reference, args.tolerance)
    if problems:
        print("Does NOT reproduce the reference evaluation:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(
        f"Reproduces the reference evaluation ({args.reference}): same weights, every test F1 "
        f"within ±{args.tolerance:g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
