"""Command-line entry point: python -m sonar_pipeline <command>."""

import argparse
import sys
from pathlib import Path

from sonar_pipeline import __version__

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
        "--output-dir",
        type=Path,
        default=Path("datasets/processed"),
        help="Folder for the .npz files, index.csv and features_meta.json "
        "(default: %(default)s).",
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

    summary = run_features(args.data_dir, args.split, args.output_dir)
    total = sum(summary.counts.values())
    print(f"Features: {summary.n_recordings} recordings, {total} segments of 3 s.")
    classes = sorted({c for c, _ in summary.counts})
    print(f"  {'class':<14}" + "".join(f"{s:>7}" for s in SPLITS) + f"{'total':>7}")
    for c in classes:
        row = [summary.counts[(c, s)] for s in SPLITS]
        print(f"  {c:<14}" + "".join(f"{n:>7}" for n in row) + f"{sum(row):>7}")
    print(f"Written to {summary.output_dir}")
    return 0


HANDLERS = {"qc": cmd_qc, "features": cmd_features}


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
