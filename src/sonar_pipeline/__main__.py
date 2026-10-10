"""Command-line entry point: python -m sonar_pipeline <command>."""

import argparse
import sys

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
    for name, (help_text, _phase) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    phase = COMMANDS[args.command][1]
    print(f"'{args.command}' is not implemented yet (phase {phase}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
