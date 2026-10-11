"""report tests (F4-T03): the report is built only from versioned files, without scoring test."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_evaluate import run, trained  # noqa: F401  (fixture)

from sonar_pipeline.__main__ import main
from sonar_pipeline.report import FIGURE_GROUPS, collect_figures, num

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_results_for_report", ROOT / "tools" / "summarize_results.py"
)
summarize_results = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summarize_results
spec.loader.exec_module(summarize_results)

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32


def write_manifests(folder: Path) -> Path:
    folder.mkdir(parents=True)
    rows = [
        ("Cargo/1.wav", "Cargo", "Cargo/A", "train", 120.0),
        ("Passengership/2.wav", "Passengership", "Passengership/B", "val", 60.0),
        ("Tanker/3.wav", "Tanker", "Tanker/C", "test", 30.0),
        ("Tug/4.wav", "Tug", "Tug/D", "train", 90.0),
    ]
    with (folder / "vessels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wav_path", "wav_duration_s"])
        w.writerows((r[0], r[4]) for r in rows)
    with (folder / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wav_path", "vessel_class", "vessel_group", "split"])
        w.writerows(r[:4] for r in rows)
    return folder


@pytest.fixture
def versioned(trained, tmp_path):  # noqa: F811
    """docs/results with two summaries of a toy evaluation, and toy manifests."""
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    results = tmp_path / "docs" / "results"
    for n in (1, 2):
        summarize_results.main(
            [
                str(tmp_path / "r"),
                str(results / f"eval{n}.json"),
                "--evaluation",
                str(n),
            ]
        )
    return results, write_manifests(tmp_path / "manifests")


def report(results, manifests, tmp_path, *extra):
    argv = ["report", "--results-dir", str(results), "--manifests-dir", str(manifests)]
    argv += ["--figures-dir", str(tmp_path / "docs" / "figures")]
    return main([*argv, "--output", str(tmp_path / "docs" / "report.md"), *extra])


def test_report_numbers_come_from_the_summaries(versioned, tmp_path, monkeypatch):
    results, manifests = versioned

    def no_scoring(*args, **kwargs):
        raise AssertionError("report must not score test")

    monkeypatch.setattr("sonar_pipeline.evaluate.evaluate_split", no_scoring)
    monkeypatch.setattr("sonar_pipeline.dataset.load_split", no_scoring)
    assert report(results, manifests, tmp_path) == 0
    text = (tmp_path / "docs" / "report.md").read_text(encoding="utf-8")
    eval2 = json.loads((results / "eval2.json").read_text(encoding="utf-8"))
    for model in eval2["models"].values():
        assert num(model["test"]["segment"]["f1_macro"]) in text
    chance = eval2["models"]["baseline"]["test"]["segment"]["chance_f1_macro"]
    assert f"*{num(chance)}*" in text
    assert "## 5. Limitaciones" in text and "DT-005" in text and "DT-007" in text
    assert "| Cargo | 2,0 min · 1 b · 1 g |" in text  # data table from the manifests
    assert (tmp_path / "docs" / "figures" / "data_minutes.png").is_file()


def test_report_is_reproducible(versioned, tmp_path):
    results, manifests = versioned
    assert report(results, manifests, tmp_path) == 0
    first = (tmp_path / "docs" / "report.md").read_bytes()
    assert report(results, manifests, tmp_path) == 0
    assert (tmp_path / "docs" / "report.md").read_bytes() == first


def test_collect_figures_copies_every_group(tmp_path):
    source = tmp_path / "outputs" / "figures"
    for names in FIGURE_GROUPS.values():
        for name in names:
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_bytes(PNG)
    copied = collect_figures(source, tmp_path / "docs" / "figures")
    assert len(copied) == sum(len(v) for v in FIGURE_GROUPS.values())
    assert (
        tmp_path / "docs" / "figures" / "eval1" / "results_f1.png"
    ).read_bytes() == PNG
    assert (tmp_path / "docs" / "figures" / "signal" / "example_tug.png").is_file()


def test_collect_figures_fails_on_a_missing_figure(tmp_path):
    with pytest.raises(FileNotFoundError, match="example_cargo.png"):
        collect_figures(tmp_path, tmp_path / "out")


def test_spanish_number_format():
    assert num(0.3739) == "0,374" and num(None) == "—" and num(95.38, 1) == "95,4"
