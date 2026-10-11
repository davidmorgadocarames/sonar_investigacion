"""Command-line entry point: python -m sonar_pipeline <command>."""

import argparse
import sys
from pathlib import Path

from sonar_pipeline import __version__

# Models of `train` (D-018); duration is the DT-005 check.
TRAIN_MODELS = ("duration", "baseline", "cnn", "cnn-aug")

# Subcommand -> (help text, phase that implements it).
COMMANDS = {
    "qc": ("Run quality control on raw WAV files.", "F1"),
    "features": ("Preprocess audio and compute LOFAR and DEMON features.", "F2"),
    "train": ("Train the baseline and CNN classifiers.", "F3"),
    "evaluate": ("Evaluate trained models on the test split.", "F3"),
    "report": ("Build the results report with figures.", "F4"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sonar_pipeline",
        description="Passive sonar pipeline on DeepShip (Cargo, Passengership, Tanker, Tug).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subcommands = {
        name: subparsers.add_parser(name, help=help_text, description=help_text)
        for name, (help_text, _phase) in COMMANDS.items()
    }

    qc = subcommands["qc"]
    qc.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/raw/DeepShip"),
        help="Folder with one subfolder per class (default: %(default)s).",
    )
    qc.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qc/qc.csv"),
        help="CSV report, one row per WAV (default: %(default)s).",
    )

    features = subcommands["features"]
    features.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/raw/DeepShip"),
        help="Folder with one subfolder per class (default: %(default)s).",
    )
    features.add_argument(
        "--split",
        type=Path,
        default=Path("datasets/manifests/split.csv"),
        help="Split manifest, one row per WAV (default: %(default)s).",
    )
    features.add_argument(
        "--vessels",
        type=Path,
        default=Path("datasets/manifests/vessels.csv"),
        help="Vessel table with the SHA-256 of every WAV (default: %(default)s).",
    )
    features.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed"),
        help="Folder for the .npz files, index.csv and features_meta.json "
        "(default: %(default)s).",
    )
    features.add_argument(
        "--figures",
        action="store_true",
        help="Also plot the LOFARgram and DEMON of the longest train recording of each "
        "class.",
    )
    features.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Folder for the figures (default: %(default)s).",
    )

    train = subcommands["train"]
    train.add_argument(
        "--model",
        choices=TRAIN_MODELS,
        required=True,
        help="duration: DT-005 check (recording duration only, scored on val); "
        "baseline, cnn, cnn-aug: the models of D-018.",
    )
    train.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("datasets/processed"),
        help="Features written by `features` (default: %(default)s).",
    )
    train.add_argument(
        "--vessels",
        type=Path,
        default=Path("datasets/manifests/vessels.csv"),
        help="Vessel table with the duration of every WAV (default: %(default)s).",
    )
    train.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Folder for the JSON results (default: %(default)s).",
    )
    return parser


def cmd_qc(args: argparse.Namespace) -> int:
    from sonar_pipeline.qc import run_qc

    results = run_qc(args.data_dir, args.output)
    flagged = [r for r in results if r.flagged]
    print(f"QC: {len(results)} WAV files checked, {len(flagged)} flagged.")
    for r in flagged:
        flags = [k for k, v in vars(r).items() if k.startswith("flag_") and v]
        print(f"  {r.path}: {', '.join(flags)}")
    print(f"Report written to {args.output}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from sonar_pipeline.features import SPLITS, run_features
    from sonar_pipeline.params import SEGMENT_S

    figures_dir = args.figures_dir if args.figures else None
    summary = run_features(
        args.data_dir, args.split, args.vessels, args.output_dir, figures_dir
    )
    total = sum(summary.counts.values())
    print(
        f"Features: {summary.n_recordings} recordings, {total} segments "
        f"of {SEGMENT_S:g} s."
    )
    classes = sorted({c for c, _ in summary.counts})
    print(f"  {'class':<14}" + "".join(f"{s:>7}" for s in SPLITS) + f"{'total':>7}")
    for c in classes:
        row = [summary.counts[(c, s)] for s in SPLITS]
        print(f"  {c:<14}" + "".join(f"{n:>7}" for n in row) + f"{sum(row):>7}")
    print(f"Written to {summary.output_dir}")
    for path in summary.figures:
        print(f"Figure: {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.model != "duration":
        print(
            f"'train --model {args.model}' is not implemented yet (F3).",
            file=sys.stderr,
        )
        return 1
    from sonar_pipeline.dataset import CLASSES
    from sonar_pipeline.duration_check import run_duration_check

    output = args.results_dir / "dt005_duration.json"
    result = run_duration_check(args.processed_dir, args.vessels, output)
    val = result["val"]
    print(
        f"DT-005 check: logistic regression on recording duration, fitted in "
        f"{result['training']['seconds']:.2f} s; scored on val (test not used)."
    )
    for level in ("segment", "recording"):
        s = val[level]
        lo, hi = s["ci95"]["f1_macro"]
        print(
            f"  {level:<9} n={s['n']:<4} F1 macro {s['f1_macro']:.3f} (95 % CI {lo:.3f}-{hi:.3f})"
        )
        print(
            "            "
            + "  ".join(f"{c} {s['f1_per_class'][c]:.2f}" for c in CLASSES)
        )
    for era, scores in val["within_era"].items():
        a, b = scores["classes"]
        for level in ("segment", "recording"):
            s = scores[level]
            print(
                f"  {era} {a} vs {b} ({level}): F1 macro {s['f1_macro']:.3f}, n={s['n']}"
            )
    print(
        f"  chance F1 macro ~ {result['chance_f1_macro']:.2f} (4 classes), 0.5 within an era"
    )
    print(f"Written to {output}")
    return 0


HANDLERS = {"qc": cmd_qc, "features": cmd_features, "train": cmd_train}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command in HANDLERS:
        return HANDLERS[args.command](args)
    phase = COMMANDS[args.command][1]
    print(f"'{args.command}' is not implemented yet (phase {phase}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
