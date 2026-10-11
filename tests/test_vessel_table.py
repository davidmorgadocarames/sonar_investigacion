"""Tests for tools/build_vessel_table.py with small synthetic metafiles and WAVs."""

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_vessel_table.py"
spec = importlib.util.spec_from_file_location("build_vessel_table", TOOL)
bvt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = (
    bvt  # dataclasses look the module up while it is being executed
)
spec.loader.exec_module(bvt)

# Low sample rate keeps the synthetic files tiny; durations stay exact.
FS = 100


def write_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(round(seconds * FS), dtype=np.float32)
    samples[::7] = 0.01
    wavfile.write(path, FS, samples)


def meta_line(row_id, name, duration, field7="1-2"):
    # Same shape as DeepShip: trailing comma and CRLF.
    return f"{row_id},60,{name},20160505,010203,{duration},{field7},\r\n"


def make_class(
    data_dir: Path, vessel_class: str, lines: list[str], wavs: dict[int, float]
) -> None:
    class_dir = data_dir / vessel_class
    class_dir.mkdir(parents=True, exist_ok=True)
    (class_dir / f"{vessel_class.lower()}-metafile").write_bytes(
        "".join(lines).encode()
    )
    for number, seconds in wavs.items():
        write_wav(class_dir / f"{number}.wav", seconds)


def by_number(table, vessel_class):
    return {int(r["wav_number"]): r for r in table if r["vessel_class"] == vessel_class}


def test_join_by_id(tmp_path):
    lines = [meta_line(i, f"SHIP {i}", 10 + i) for i in range(1, 6)]
    make_class(tmp_path, "Tug", lines, {2: 12, 4: 14})
    rows = by_number(bvt.build_table(tmp_path), "Tug")
    for n in (2, 4):
        assert rows[n]["match_source"] == "id"
        assert rows[n]["metafile_id"] == str(n)
        assert rows[n]["metafile_line"] == str(n)
        assert rows[n]["vessel_name_raw"] == f"SHIP {n}"
        assert rows[n]["duration_match"] == "true"


def test_duration_match_breaks_ties_by_closest_id(tmp_path):
    # WAV 5 lasts 20 s; row 5 lasts 50 s; rows 4 and 7 both last 20 s -> 4 is closer.
    durations = {1: 31, 2: 32, 3: 33, 4: 20, 5: 50, 6: 36, 7: 20, 8: 38}
    lines = [meta_line(i, f"SHIP {i}", d) for i, d in durations.items()]
    make_class(tmp_path, "Cargo", lines, {5: 20})
    row = by_number(bvt.build_table(tmp_path), "Cargo")[5]
    assert row["match_source"] == "duration"
    assert row["metafile_id"] == "4"
    assert row["duration_match"] == "true"


def test_duration_tolerance_is_one_second(tmp_path):
    lines = [meta_line(1, "A", 100), meta_line(2, "B", 200)]
    make_class(tmp_path, "Cargo", lines, {1: 101.0, 2: 201.5})
    rows = by_number(bvt.build_table(tmp_path), "Cargo")
    assert rows[1]["match_source"] == "id"
    assert rows[2]["match_source"] == "id_unverified"


def test_order_is_preserved_over_greedy_id_match(tmp_path):
    # The Tanker 33/35 pattern: WAV 33 and 35 both last 16 s, rows 35 and 37 too. A greedy
    # "id first" join would take row 35 for WAV 35 and push WAV 33 forward to row 37, which
    # crosses the chronological order. The alignment keeps both at offset +2.
    durations = {31: 50, 32: 51, 33: 52, 34: 53, 35: 16, 36: 55, 37: 16, 38: 57}
    lines = [meta_line(i, f"SHIP {i}", d) for i, d in durations.items()]
    make_class(tmp_path, "Tanker", lines, {33: 16, 35: 16})
    rows = by_number(bvt.build_table(tmp_path), "Tanker")
    assert rows[33]["metafile_id"] == "35"
    assert rows[35]["metafile_id"] == "37"
    assert {rows[n]["match_source"] for n in (33, 35)} == {"duration"}


def test_unmatched_duration_keeps_id_row_unverified(tmp_path):
    lines = [
        meta_line(1, "A", 10),
        meta_line(2, "COLUMBIA ", 252),
        meta_line(3, "C", 30),
    ]
    make_class(tmp_path, "Passengership", lines, {1: 10, 2: 6, 3: 30})
    rows = by_number(bvt.build_table(tmp_path), "Passengership")
    assert rows[2]["match_source"] == "id_unverified"
    assert rows[2]["metafile_id"] == "2"
    assert rows[2]["duration_match"] == "false"
    assert rows[2]["duration_diff_s"] == "-246.000000"


def test_restarted_numbering_uses_best_block(tmp_path):
    # Block 1 matches the WAVs; block 2 restarts at id 1 with other durations.
    block1 = [meta_line(i, f"FIRST {i}", 10 * i) for i in range(1, 5)]
    block2 = [meta_line(i, f"SECOND {i}", 7 * i + 3) for i in range(1, 5)]
    make_class(tmp_path, "Tanker", block1 + ["\r\n"] + block2, {2: 20, 3: 30})
    rows = by_number(bvt.build_table(tmp_path), "Tanker")
    assert {rows[n]["metafile_block"] for n in (2, 3)} == {"1"}
    assert rows[2]["vessel_name_raw"] == "FIRST 2"
    parsed = bvt.parse_metafile(tmp_path / "Tanker" / "tanker-metafile")
    assert [r.block for r in parsed] == [1] * 4 + [2] * 4
    assert parsed[4].line == 6  # the blank line still counts as a physical line


def test_ambiguous_alignment_fails(tmp_path):
    # WAV 5 (20 s) is equally close to rows 4 and 6, both 20 s.
    durations = {3: 30, 4: 20, 5: 50, 6: 20, 7: 70}
    lines = [meta_line(i, f"SHIP {i}", d) for i, d in durations.items()]
    make_class(tmp_path, "Cargo", lines, {5: 20})
    with pytest.raises(bvt.AlignmentError, match="ambiguous"):
        bvt.build_table(tmp_path)


def test_malformed_metafile_line_fails(tmp_path):
    make_class(tmp_path, "Tug", ["1,52,X,20160505,010203,10,1-2,extra\r\n"], {1: 10})
    with pytest.raises(bvt.MetafileError):
        bvt.build_table(tmp_path)


def test_field7_and_raw_name_are_kept_verbatim(tmp_path):
    make_class(
        tmp_path, "Tug", [meta_line(1, "KIRKEHOLMEN ", 10, field7="1488-466")], {1: 10}
    )
    row = bvt.build_table(tmp_path)[0]
    assert row["vessel_name_raw"] == "KIRKEHOLMEN "
    assert row["vessel_group"] == "Tug/KIRKEHOLMEN"
    assert row["field7"] == "1488-466"
    assert row["time"] == "010203"


@pytest.mark.parametrize(
    ("raw", "group"),
    [
        ("QUEEN OF", "Passengership/QUEEN OF"),
        ("QUEEN OF NEW WESTMINSTER", "Passengership/QUEEN OF"),
        ("  queen  of   cowichan ", "Passengership/QUEEN OF"),
        ("MAYNE QUEEN", "Passengership/MAYNE QUEEN"),
        (" columbia\t", "Passengership/COLUMBIA"),
        ("Spirit  of  X", "Passengership/SPIRIT OF X"),
    ],
)
def test_vessel_group_normalisation(raw, group):
    assert bvt.vessel_group("Passengership", raw) == group


def test_doubtful_cases_share_a_group_and_groups_never_span_classes(tmp_path):
    lines = [
        meta_line(1, "QUEEN OF", 37),
        meta_line(2, "QUEEN OF SURREY", 48),
        meta_line(3, "COLUMBIA ", 252),  # WAV 3 lasts 6 s -> id_unverified
        meta_line(4, "COLUMBIA", 54),
    ]
    make_class(tmp_path, "Passengership", lines, {1: 37, 2: 48, 3: 6, 4: 54})
    make_class(tmp_path, "Tanker", [meta_line(1, "COLUMBIA", 10)], {1: 10})
    table = bvt.build_table(tmp_path)
    rows = by_number(table, "Passengership")
    assert (
        rows[1]["vessel_group"] == rows[2]["vessel_group"] == "Passengership/QUEEN OF"
    )
    assert rows[3]["match_source"] == "id_unverified"
    assert (
        rows[3]["vessel_group"] == rows[4]["vessel_group"] == "Passengership/COLUMBIA"
    )
    assert by_number(table, "Tanker")[1]["vessel_group"] == "Tanker/COLUMBIA"
    classes_per_group = {}
    for r in table:
        classes_per_group.setdefault(r["vessel_group"], set()).add(r["vessel_class"])
    assert all(len(c) == 1 for c in classes_per_group.values())


def test_output_is_deterministic_sorted_lf_utf8(tmp_path):
    data = tmp_path / "data"
    make_class(
        data,
        "Tug",
        [meta_line(i, f"T{i}", 10 + i) for i in range(1, 12)],
        {2: 12, 10: 20},
    )
    make_class(
        data,
        "Cargo",
        [meta_line(i, f"C{i}", 20 + i) for i in range(1, 4)],
        {3: 23, 1: 21},
    )
    out1, out2 = tmp_path / "a" / "vessels.csv", tmp_path / "b" / "vessels.csv"
    assert bvt.main(["--data-dir", str(data), "--output", str(out1)]) == 0
    assert bvt.main(["--data-dir", str(data), "--output", str(out2)]) == 0
    content = out1.read_bytes()
    assert content == out2.read_bytes()
    assert b"\r" not in content
    content.decode("utf-8")
    with out1.open(newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert list(table[0]) == bvt.FIELDS
    assert [r["wav_path"] for r in table] == [
        "Cargo/1.wav",
        "Cargo/3.wav",
        "Tug/2.wav",
        "Tug/10.wav",
    ]
    wav = data / "Tug" / "10.wav"
    assert table[3]["sha256"] == hashlib.sha256(wav.read_bytes()).hexdigest()
    assert table[3]["n_samples"] == str(20 * FS)
