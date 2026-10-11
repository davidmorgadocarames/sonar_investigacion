"""DT-002 checks on the real vessel table and, when present, on the real DeepShip WAVs.

``datasets/manifests/vessels.csv`` is versioned, so its tests run whenever ``datasets/`` is there
(skipped with a bare ``docker run``). The WAV tests need ``datasets/raw/DeepShip`` (mounted as a volume in Docker) and are skipped if it is missing.
The WAVs and metafiles are read here directly, independently of tools/build_vessel_table.py.
"""

import csv
import hashlib
from pathlib import Path

import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
VESSELS = ROOT / "datasets" / "manifests" / "vessels.csv"
DATA_DIR = ROOT / "datasets" / "raw" / "DeepShip"

TOLERANCE_S = 1.0
VERIFIED = {"id", "duration"}
UNVERIFIED = {"Passengership/30.wav", "Passengership/37.wav"}


@pytest.fixture(scope="module")
def table():
    if not VESSELS.parent.is_dir():
        pytest.skip(
            f"{VESSELS.parent} not found: datasets/ is not mounted (use docker compose)"
        )
    assert VESSELS.is_file(), f"{VESSELS} is versioned and must exist"
    with VESSELS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_verified_rows_match_metafile_duration(table):
    verified = [r for r in table if r["match_source"] in VERIFIED]
    assert len(table) == 63
    assert len(verified) == 61
    for r in verified:
        diff = float(r["wav_duration_s"]) - int(r["metafile_duration_s"])
        assert r["duration_match"] == "true", r["wav_path"]
        assert abs(diff) <= TOLERANCE_S, r["wav_path"]
        assert float(r["duration_diff_s"]) == pytest.approx(diff, abs=1e-6)


def test_only_unverified_rows_are_columbia(table):
    unverified = [r for r in table if r["match_source"] not in VERIFIED]
    assert {r["wav_path"] for r in unverified} == UNVERIFIED
    for r in unverified:
        assert r["match_source"] == "id_unverified"
        assert r["vessel_group"] == "Passengership/COLUMBIA"
    # D-007: they share the group with the verified COLUMBIA recording(s).
    assert any(
        r["vessel_group"] == "Passengership/COLUMBIA" and r["match_source"] in VERIFIED
        for r in table
    )


# --------------------------------------------------------------------------- real WAVs


@pytest.fixture(scope="module")
def data_dir():
    if not DATA_DIR.is_dir():
        pytest.skip(
            f"{DATA_DIR} not found: download DeepShip as described in README.md "
            "to check the WAVs against vessels.csv"
        )
    return DATA_DIR


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metafile_line(path: Path, line_no: int) -> list[str]:
    return path.read_bytes().decode("utf-8").splitlines()[line_no - 1].split(",")


def test_wav_set_matches_table(table, data_dir):
    on_disk = {p.relative_to(data_dir).as_posix() for p in data_dir.glob("*/*.wav")}
    assert on_disk == {r["wav_path"] for r in table}


def test_wav_content_and_duration_match_table(table, data_dir):
    for r in table:
        path = data_dir / r["wav_path"]
        sample_rate, samples = wavfile.read(path, mmap=True)
        n_samples = int(samples.shape[0])
        del samples
        assert int(r["sample_rate"]) == sample_rate, r["wav_path"]
        assert int(r["n_samples"]) == n_samples, r["wav_path"]
        assert r["wav_duration_s"] == f"{n_samples / sample_rate:.6f}", r["wav_path"]
        assert r["sha256"] == sha256_of(path), r["wav_path"]


def test_verified_wav_duration_matches_its_metafile_row(table, data_dir):
    """DT-002: each verified WAV lasts what its assigned metafile line says (within 1 s)."""
    checked = 0
    for r in table:
        fields = metafile_line(data_dir / r["metafile"], int(r["metafile_line"]))
        assert int(fields[0]) == int(r["metafile_id"]), r["wav_path"]
        assert fields[2] == r["vessel_name_raw"], r["wav_path"]
        assert int(fields[5]) == int(r["metafile_duration_s"]), r["wav_path"]
        if r["match_source"] in VERIFIED:
            sample_rate, samples = wavfile.read(data_dir / r["wav_path"], mmap=True)
            seconds = samples.shape[0] / sample_rate
            del samples
            assert abs(seconds - int(fields[5])) <= TOLERANCE_S, r["wav_path"]
            checked += 1
    assert checked == 61
