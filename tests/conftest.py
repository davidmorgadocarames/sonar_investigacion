"""Shared pytest options."""

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--figures-dir",
        type=Path,
        default=None,
        help="Keep the synthetic validation figures in this folder (D-016), "
        "e.g. outputs/figures. Without it they go to a temporary folder.",
    )
