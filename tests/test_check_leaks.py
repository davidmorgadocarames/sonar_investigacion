"""Tests for tools/check_leaks.py with small synthetic split and vessel manifests."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_leaks.py"
spec = importlib.util.spec_from_file_location("check_leaks", TOOL)
cl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cl
spec.loader.exec_module(cl)

# (wav_path, vessel_group, sha256 seed, duration_s)
VESSELS = [
    ("Cargo/1.wav", "Cargo/ALPHA", "a1", 3600),
    ("Cargo/2.wav", "Cargo/ALPHA", "a2", 1800),
    ("Cargo/3.wav", "Cargo/BRAVO", "b3", 900),
    ("Cargo/4.wav", "Cargo/CHARLIE", "c4", 900),
    ("Tug/1.wav", "Tug/DELTA", "d1", 600),
    ("Tug/2.wav", "Tug/ECHO", "e2", 600),
    ("Tug/3.wav", "Tug/FOXTROT", "f3", 600),
]
CLEAN_SPLIT = {
    "Cargo/1.wav": "train",
    "Cargo/2.wav": "train",
    "Cargo/3.wav": "val",
    "Cargo/4.wav": "test",
    "Tug/1.wav": "train",
    "Tug/2.wav": "val",
    "Tug/3.wav": "test",
}


def vessel_rows():
    return [
        {
            "wav_path": path,
            "vessel_class": path.split("/")[0],
            "vessel_group": group,
            "sha256": seed * 32,
            "wav_duration_s": f"{seconds:.6f}",
        }
        for path, group, seed, seconds in VESSELS
    ]


def split_rows(assignment=CLEAN_SPLIT):
    groups = {path: group for path, group, _, _ in VESSELS}
    return [
        {
            "wav_path": path,
            "vessel_class": path.split("/")[0],
            "vessel_group": groups.get(path, path.split("/")[0] + "/UNKNOWN"),
            "split": split,
        }
        for path, split in assignment.items()
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(tmp_path, split, vessels=None):
    split_csv = write_csv(tmp_path / "split.csv", split)
    vessels_csv = write_csv(tmp_path / "vessels.csv", vessels or vessel_rows())
    return cl.main([str(split_csv), "--vessels", str(vessels_csv)])


def test_clean_split_passes(tmp_path, capsys):
    assert cl.check_split(split_rows(), vessel_rows()) == []
    assert run(tmp_path, split_rows()) == 0
    out = capsys.readouterr().out
    assert "OK: no leaks" in out
    # Cargo train: ALPHA (1 h + 0.5 h) -> 2 WAV, 1 group, 1.5 h
    assert any(
        line.split() == ["Cargo", "2", "WAV", "1", "groups", "1.500", "h"]
        for line in out.splitlines()
    )
    assert any(
        line.startswith("all ") and " 7 WAV" in line for line in out.splitlines()
    )


def test_vessel_leak_fails(tmp_path, capsys):
    split = split_rows(
        {**CLEAN_SPLIT, "Cargo/2.wav": "test"}
    )  # ALPHA in train and test
    errors = cl.check_split(split, vessel_rows())
    assert errors == ["vessel leak: Cargo/ALPHA is in train/test"]
    assert run(tmp_path, split) == 1
    assert "vessel leak: Cargo/ALPHA" in capsys.readouterr().err


def test_vessel_leak_is_found_on_reference_group(tmp_path):
    # The split hides the leak by declaring another group; vessels.csv reveals it.
    split = split_rows({**CLEAN_SPLIT, "Cargo/2.wav": "test"})
    split[1]["vessel_group"] = "Cargo/ALPHA 2"
    errors = cl.check_split(split, vessel_rows())
    assert "vessel leak: Cargo/ALPHA is in train/test" in errors
    assert any(e.startswith("vessel_group mismatch: Cargo/2.wav") for e in errors)
    assert run(tmp_path, split) == 1


def test_recording_leak_same_path_twice_fails(tmp_path):
    split = split_rows() + [
        {
            "wav_path": "Tug/3.wav",
            "vessel_class": "Tug",
            "vessel_group": "Tug/FOXTROT",
            "split": "train",
        }
    ]
    errors = cl.check_split(split, vessel_rows())
    assert "recording leak: Tug/3.wav appears 2 times (splits train/test)" in errors
    assert "vessel leak: Tug/FOXTROT is in train/test" in errors
    assert run(tmp_path, split) == 1


def test_recording_listed_twice_in_same_split_fails():
    split = split_rows() + [split_rows()[0]]
    errors = cl.check_split(split, vessel_rows())
    assert errors == ["recording leak: Cargo/1.wav appears 2 times (splits train)"]


def test_duplicate_sha256_fails(tmp_path):
    # Tug/2.wav has the same content as Tug/1.wav: one recording under two paths.
    vessels = vessel_rows()
    vessels[5]["sha256"] = vessels[4]["sha256"]
    errors = cl.check_split(split_rows(), vessels)
    assert any(
        e.startswith("recording leak: same content")
        and "['Tug/1.wav', 'Tug/2.wav']" in e
        and "train/val" in e
        for e in errors
    )
    assert any(e.startswith("vessels.csv: same content") for e in errors)
    assert run(tmp_path, split_rows(), vessels) == 1


def test_group_mismatch_fails(tmp_path):
    split = split_rows()
    split[3]["vessel_group"] = "Cargo/BRAVO"  # Cargo/4.wav is CHARLIE
    errors = cl.check_split(split, vessel_rows())
    assert (
        "vessel_group mismatch: Cargo/4.wav has 'Cargo/BRAVO' in the split "
        "but 'Cargo/CHARLIE' in vessels.csv"
    ) in errors
    assert "vessel leak: Cargo/BRAVO is in val/test" in errors
    assert run(tmp_path, split) == 1


def test_class_mismatch_fails():
    split = split_rows()
    split[4]["vessel_class"] = "Cargo"
    errors = cl.check_split(split, vessel_rows())
    assert errors == [
        "vessel_class mismatch: Tug/1.wav has 'Cargo' in the split but 'Tug' in vessels.csv"
    ]


def test_missing_wav_fails(tmp_path):
    assignment = {k: v for k, v in CLEAN_SPLIT.items() if k != "Cargo/3.wav"}
    errors = cl.check_split(split_rows(assignment), vessel_rows())
    assert errors == ["missing WAV: Cargo/3.wav from vessels.csv is not in the split"]
    assert run(tmp_path, split_rows(assignment)) == 1


def test_unknown_wav_fails(tmp_path):
    split = split_rows({**CLEAN_SPLIT, "Cargo/99.wav": "train"})
    errors = cl.check_split(split, vessel_rows())
    assert errors == ["unknown WAV: Cargo/99.wav is not in vessels.csv"]
    assert run(tmp_path, split) == 1


@pytest.mark.parametrize("value", ["Train", "validation", "", "dev"])
def test_invalid_split_value_fails(value):
    split = split_rows()
    split[6]["split"] = value
    errors = cl.check_split(split, vessel_rows())
    assert errors == [
        f"line 8: Tug/3.wav: invalid split {value!r} (expected one of train, val, test)"
    ]


def test_missing_split_file_or_column_exits_2(tmp_path, capsys):
    vessels_csv = write_csv(tmp_path / "vessels.csv", vessel_rows())
    missing = tmp_path / "nope.csv"
    assert cl.main([str(missing), "--vessels", str(vessels_csv)]) == 2
    assert "F1-T06c" in capsys.readouterr().err
    no_split_column = [
        {k: v for k, v in r.items() if k != "split"} for r in split_rows()
    ]
    split_csv = write_csv(tmp_path / "split.csv", no_split_column)
    assert cl.main([str(split_csv), "--vessels", str(vessels_csv)]) == 2
    assert "missing columns ['split']" in capsys.readouterr().err


def test_edit_distance():
    assert cl.edit_distance("KITTEN", "SITTING") == 3
    assert cl.edit_distance("", "ABC") == 3
    assert cl.edit_distance("SAME", "SAME") == 0


def test_similar_names_criteria(tmp_path, capsys):
    groups = [
        ("Passengership/1.wav", "Passengership/SPIRIT OF"),
        ("Passengership/2.wav", "Passengership/SPIRIT OF VANCOUVER ISLAND"),
        ("Passengership/3.wav", "Passengership/SPIRIT OF VANCOUVER ISLAND"),
        ("Tanker/1.wav", "Tanker/LYNDA VICTORY"),
        ("Tanker/2.wav", "Tanker/LINDA VICTORIA"),
        ("Tanker/3.wav", "Tanker/UNRELATED"),
        ("Tug/1.wav", "Tug/UNRELATED"),
    ]
    vessels = [
        {
            "wav_path": p,
            "vessel_class": p.split("/")[0],
            "vessel_group": g,
            "sha256": str(i) * 64,
            "wav_duration_s": "1.0",
        }
        for i, (p, g) in enumerate(groups)
    ]
    pairs = {(p["group_a"], p["group_b"]): p for p in cl.similar_names(vessels)}
    assert len(pairs) == 3
    spirit = pairs[
        ("Passengership/SPIRIT OF", "Passengership/SPIRIT OF VANCOUVER ISLAND")
    ]
    assert spirit["criteria"] == "prefix+shared_word:OF SPIRIT"
    assert (spirit["scope"], spirit["wav_a"], spirit["wav_b"]) == (
        "same_class",
        "1",
        "2",
    )
    victory = pairs[("Tanker/LINDA VICTORIA", "Tanker/LYNDA VICTORY")]
    assert victory["criteria"] == "edit_distance<=3"
    cross = pairs[("Tanker/UNRELATED", "Tug/UNRELATED")]
    assert (cross["scope"], cross["criteria"]) == (
        "cross_class",
        "same_name+shared_word:UNRELATED",
    )
    vessels_csv = write_csv(tmp_path / "vessels.csv", vessels)
    assert cl.main(["--similar-names", "--vessels", str(vessels_csv)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "scope,group_a,wav_a,group_b,wav_b,criteria"
    assert len(out) == 4
