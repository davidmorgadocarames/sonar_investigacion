"""DT-004: rebuilding vessels.csv from the real DeepShip WAVs gives the versioned table, byte for byte.

If the rules of tools/build_vessel_table.py change (grouping, alignment) and the table is not
regenerated, this test fails. It needs ``datasets/raw/DeepShip`` (mounted as a volume in Docker)
and is skipped if the folder or its WAVs are missing.
"""

import difflib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VESSELS = ROOT / "datasets" / "manifests" / "vessels.csv"
DATA_DIR = ROOT / "datasets" / "raw" / "DeepShip"

TOOL = ROOT / "tools" / "build_vessel_table.py"
spec = importlib.util.spec_from_file_location("build_vessel_table", TOOL)
bvt = importlib.util.module_from_spec(spec)
# dataclasses look the module up while it is being executed
sys.modules[spec.name] = bvt
spec.loader.exec_module(bvt)

REGENERATE = (
    "docker compose run --rm --entrypoint python pipeline tools/build_vessel_table.py"
)


def divergence_message(expected: bytes, actual: bytes, max_lines: int = 40) -> str:
    diff = list(
        difflib.unified_diff(
            expected.decode("utf-8").splitlines(),
            actual.decode("utf-8").splitlines(),
            fromfile="datasets/manifests/vessels.csv (versioned)",
            tofile="rebuilt by tools/build_vessel_table.py",
            lineterm="",
        )
    )
    shown = "\n".join(diff[:max_lines])
    if len(diff) > max_lines:
        shown += f"\n... ({len(diff) - max_lines} more diff lines)"
    if not diff:
        shown = (
            "(same lines; the bytes differ: line endings, encoding or trailing newline)"
        )
    return (
        "DT-004: tools/build_vessel_table.py no longer reproduces the versioned vessels.csv.\n"
        f"{shown}\n"
        f"To regenerate it: {REGENERATE}\n"
        "WARNING: regenerating changes vessel groups and therefore the train/val/test split "
        "(split.csv). Do not commit a regenerated table without David's decision "
        "(rule 2, DECISIONS.md)."
    )


def test_rebuilt_vessel_table_matches_versioned(tmp_path):
    if not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*/*.wav")):
        pytest.skip(
            f"{DATA_DIR} not found or without WAVs: download DeepShip as described in "
            "README.md to check that vessels.csv can be rebuilt from the audio"
        )
    assert VESSELS.is_file(), f"{VESSELS} is versioned and must exist"
    output = tmp_path / "vessels.csv"
    assert bvt.main(["--data-dir", str(DATA_DIR), "--output", str(output)]) == 0
    expected = VESSELS.read_bytes()
    actual = output.read_bytes()
    assert actual == expected, divergence_message(expected, actual)
