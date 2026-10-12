"""Command-line entry point: python -m sonar_pipeline <command>."""

import argparse
import sys
from pathlib import Path

from sonar_pipeline import __version__

# Models of `train` (D-018); duration is the DT-005 check.
TRAIN_MODELS = ("duration", "baseline", "cnn", "cnn-aug")
# Models of round 3 (F5, D-024); kept here so --help does not import torch.
ROUND3_MODELS = (
    "baseline",
    "cnn",
    "cnn_aug",
    "hybrid",
    "hybrid_aug",
    "transformer",
    "transformer_aug",
)

# Subcommand -> (help text, phase that implements it).
COMMANDS = {
    "qc": ("Run quality control on raw WAV files.", "F1"),
    "features": ("Preprocess audio and compute LOFAR and DEMON features.", "F2"),
    "train": (
        "Train classifiers; --round 3: all 7 models (5x5 CV).",
        "F3",
    ),
    "evaluate": (
        "Evaluate models; --round 3: checks and summaries.",
        "F3",
    ),
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
        help="duration: check whether the recording duration alone predicts the class "
        "(scored on val); baseline, cnn, cnn-aug: the models of the earlier fixed-split "
        "evaluation. Not with --round.",
    )
    train.add_argument(
        "--round",
        type=int,
        choices=(3,),
        help="3: the 7 models under 5 x 5 cross-validation by vessel, "
        "one after another; resumes finished trainings. Not with --model.",
    )
    train.add_argument(
        "--only",
        nargs="+",
        choices=ROUND3_MODELS,
        metavar="MODEL",
        help=f"--round 3: run only these models ({', '.join(ROUND3_MODELS)}).",
    )
    train.add_argument(
        "--measure",
        action="store_true",
        help="--round 3: only measure the epoch time and GPU memory of every network "
        "configuration on the first partition (2 epochs each; writes only measure.json and run.log).",
    )
    train.add_argument(
        "--splits",
        type=Path,
        default=Path("datasets/manifests/cv_splits.csv"),
        help="--round 3: the 25 partitions (default: %(default)s).",
    )
    train.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/round3"),
        help="--round 3: one JSON per training and per model (default: %(default)s).",
    )
    train.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/results/round3.json"),
        help="--round 3: versioned summary with the comparisons (default: %(default)s).",
    )
    train.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Network models: early-stopping patience in epochs (default: %(default)s).",
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
        "--device",
        default="auto",
        help="Network models: auto (CUDA if available, else CPU), cpu or cuda (default: auto).",
    )
    train.add_argument(
        "--max-epochs",
        type=int,
        default=60,
        help="Network models: maximum epochs per run (default: %(default)s).",
    )
    train.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap resamples of recordings for the intervals (default: 2000; 200 per "
        "partition with --round 3).",
    )
    train.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Folder for the weights and their JSON (default: %(default)s).",
    )
    train.add_argument(
        "--permutation-test",
        type=int,
        default=0,
        metavar="N",
        help="baseline only: instead of training, run N permutations of the classes between "
        "vessel groups with the saved C (after the model is committed).",
    )
    train.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Folder for the JSON results (default: %(default)s).",
    )

    evaluate = subcommands["evaluate"]
    evaluate.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("datasets/processed"),
        help="Features written by `features` (default: %(default)s).",
    )
    evaluate.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Versioned weights and their JSON (default: %(default)s).",
    )
    evaluate.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Folder for metrics.json (default: outputs/results; with --round 3, docs/results, "
        "where round3.json is versioned).",
    )
    evaluate.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Folder for the result figures (default: %(default)s).",
    )
    evaluate.add_argument(
        "--device",
        default="cpu",
        help="cpu (reference, default), cuda or auto.",
    )
    evaluate.add_argument(
        "--refresh",
        action="store_true",
        help="Reload the duration check, permutation and interpretability results into metrics.json "
        "without scoring test again.",
    )
    evaluate.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the result figures from metrics.json without scoring test again.",
    )
    evaluate.add_argument(
        "--interpretability",
        action="store_true",
        help="Instead of scoring test, compute the interpretability on val with the frozen "
        "models and add it to metrics.json.",
    )
    evaluate.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/round3"),
        help="--round 3: the per-training JSON written by train --round 3 "
        "(default: %(default)s).",
    )
    evaluate.add_argument(
        "--check",
        action="store_true",
        help="--round 3: quick check (CPU, a few seconds) that the 7 representative models in "
        "models/round3/representative/ reproduce their test F1 recorded in round3.json.",
    )
    evaluate.add_argument(
        "--details",
        action="store_true",
        help="--round 3: collect speed, curves, train and val F1, per-vessel results, "
        "calibration and ROC into round3_details.json for the report (no training).",
    )
    evaluate.add_argument(
        "--round",
        type=int,
        choices=(3,),
        help="3: the cross-validation of the 7 models. With --check, rescore the 7 "
        "representative models; with --interpretability, interpretability on the val of every "
        "partition (test not used); with --details, the summaries for the report.",
    )
    evaluate.add_argument(
        "--splits",
        type=Path,
        default=Path("datasets/manifests/cv_splits.csv"),
        help="--round 3: the 25 partitions (default: %(default)s).",
    )
    evaluate.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        metavar="N",
        help="Bootstrap resamples of recordings for the intervals (default: %(default)s).",
    )

    report = subcommands["report"]
    report.add_argument(
        "--results-dir",
        type=Path,
        default=Path("docs/results"),
        help="Versioned result summaries (default: %(default)s).",
    )
    report.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Versioned figures (default: %(default)s).",
    )
    report.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("datasets/manifests"),
        help="vessels.csv and cv_splits.csv, for the data section (default: %(default)s).",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=Path("docs/report.md"),
        help="Report written in Markdown (default: %(default)s).",
    )
    report.add_argument(
        "--collect-figures",
        type=Path,
        default=None,
        metavar="DIR",
        help="First copy the figures of DIR (e.g. outputs/figures) into the figures folder.",
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
    print(f"  {'class':<14}{'segments':>9}")
    for c in classes:
        print(f"  {c:<14}{sum(summary.counts[(c, s)] for s in SPLITS):>9}")
    print(
        f"Written to {summary.output_dir}. The segments are stored in three files "
        "(features_train/val/test.npz) only to keep them; the evaluation joins them and uses "
        "the 25 partitions of datasets/manifests/cv_splits.csv (5 repetitions x 5 folds)."
    )
    for path in summary.figures:
        print(f"Figure: {path}")
    return 0


def print_val_scores(val: dict) -> None:
    from sonar_pipeline.dataset import CLASSES

    for level in ("segment", "recording"):
        s = val[level]
        lo, hi = s["ci95"]["f1_macro"]
        print(
            f"  {level:<9} n={s['n']:<4} F1 macro {s['f1_macro']:.3f} "
            f"(95 % CI {lo:.3f}-{hi:.3f})"
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


def train_duration(args: argparse.Namespace) -> int:
    from sonar_pipeline.duration_check import run_duration_check

    output = args.results_dir / "dt005_duration.json"
    result = run_duration_check(
        args.processed_dir, args.vessels, output, args.bootstrap
    )
    print(
        f"Duration check: logistic regression on recording duration, fitted in "
        f"{result['training']['seconds']:.2f} s; scored on val (test not used)."
    )
    print_val_scores(result["val"])
    val = result["val"]
    print(
        f"  measured chance F1 macro: {val['segment']['chance_f1_macro']:.3f} (segment), "
        f"{val['recording']['chance_f1_macro']:.3f} (recording)"
    )
    print(f"Written to {output}")
    return 0


def train_baseline(args: argparse.Namespace) -> int:
    from sonar_pipeline.baseline import permutation_test, train_baseline

    if args.permutation_test:
        output = args.results_dir / "permutation_test.json"
        result = permutation_test(
            args.processed_dir, args.models_dir, output, args.permutation_test
        )
        print(
            f"Permutation test (baseline, C={result['c']:g}): {result['n_permutations']} "
            f"permutations of the classes between vessel groups in {result['seconds']:.0f} s."
        )
        print(
            f"  grouped CV F1 macro: real labels {result['observed_f1_macro']:.3f}, "
            f"permuted mean {result['null_mean']:.3f} (95th percentile "
            f"{result['null_p95']:.3f}); p = {result['p_value']:.4f}"
        )
        print(f"Written to {output}")
        return 0

    report = train_baseline(args.processed_dir, args.models_dir, args.bootstrap)
    print(
        f"Baseline: logistic regression, C={report['config']['c']:g} chosen on val "
        f"(search {report['training']['search_seconds']:.1f} s, final fit "
        f"{report['training']['seconds']:.1f} s; test not used)."
    )
    for row in report["training"]["search"]:
        print(f"  C={row['c']:<6g} val F1 macro {row['val_f1_macro']:.3f}")
    print("Val scores of the chosen model:")
    print_val_scores(report["val"])
    cv = report["grouped_cv"]
    print(
        f"  grouped CV on train + val ({cv['folds']} folds by vessel): F1 macro "
        f"{cv['mean']:.3f} +- {cv['std']:.3f}"
    )
    print(f"Written to {args.models_dir / 'baseline.npz'} and baseline.json")
    return 0


def train_cnn_model(args: argparse.Namespace) -> int:
    from sonar_pipeline.cnn import train_cnn, train_cnn_aug

    options = {
        "device_name": args.device,
        "n_bootstrap": args.bootstrap,
        "max_epochs": args.max_epochs,
    }
    if args.model == "cnn-aug":
        report = train_cnn_aug(args.processed_dir, args.models_dir, **options)
    else:
        report = train_cnn(args.processed_dir, args.models_dir, name="cnn", **options)
    t = report["training"]
    c = report["config"]
    print(
        f"CNN on {t['device']}: {len(t['search'])} configurations x seed {c['seeds'][0]}, then "
        f"{len(t['seeds'])} seeds, in {t['total_seconds']:.0f} s (test not used)."
    )
    for run in t["search"]:
        print(
            f"  lr={run['lr']:g} wd={run['weight_decay']:g}: val F1 macro "
            f"{run['best_val_f1_macro']:.3f} (epoch {run['best_epoch']}/{run['epochs_run']}, "
            f"{run['seconds']:.0f} s)"
        )
    print(
        f"  chosen lr={c['lr']:g} wd={c['weight_decay']:g}; seeds "
        + ", ".join(f"{r['seed']}: {r['best_val_f1_macro']:.3f}" for r in t["seeds"])
        + f" (mean {t['seed_val_f1_mean']:.3f} +- {t['seed_val_f1_std']:.3f}); kept seed {c['seed']}"
    )
    print(
        f"  {report['cost']['parameters']} parameters, "
        f"{report['cost']['flops_per_segment'] / 1e6:.1f} MFLOPs per segment"
    )
    print("Val scores of the kept model:")
    print_val_scores(report["val"])
    if "comparison" in report:
        cmp = report["comparison"]
        print(
            f"  same seeds on val: cnn {cmp['cnn_seed_val_f1_mean']:.3f} "
            f"vs cnn-aug {cmp['cnn_aug_seed_val_f1_mean']:.3f} (mean F1 macro)"
        )
    print(f"Written to {args.models_dir / (report['model'] + '.pt')} and its .json")
    return 0


def train_round3(args: argparse.Namespace) -> int:
    from sonar_pipeline.cv import MODELS, PARTITION_BOOTSTRAP, Round3

    log_path = args.outputs_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    engine = Round3(
        args.processed_dir,
        args.splits,
        args.models_dir / "round3",
        args.outputs_dir,
        device_name=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_bootstrap=args.bootstrap or PARTITION_BOOTSTRAP,
        log=log,
    )
    log(f"Round 3 on {engine.device}: {len(engine.partitions)} partitions")
    if args.measure:
        import json

        result = engine.measure()
        path = args.outputs_dir / "measure.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        log(f"Written to {path}")
        return 0
    summary = engine.run(tuple(args.only or MODELS), args.summary)
    for name, entry in summary["models"].items():
        f1 = entry["test"]["f1_macro_segment"]
        log(
            f"  {name:<16} config {entry['chosen_config']}  test F1 (segment) "
            f"{f1['mean']:.3f} +- {f1['std']:.3f}"
        )
    for c in summary["comparisons"]:
        log(
            f"  {c['model']} vs {c['against']}: {c['mean_difference']:+.3f}, "
            f"p {c['p']:.3f}, Holm {c['p_holm']:.3f}"
        )
    for name, error in summary["failures"].items():
        log(f"  {name} FAILED: {error}")
    log(f"Written to {args.summary}")
    return 1 if summary["failures"] else 0


def cmd_train(args: argparse.Namespace) -> int:
    if (args.model is None) == (args.round is None):
        print("Give either --model or --round.", file=sys.stderr)
        return 1
    if args.round == 3:
        return train_round3(args)
    if args.only or args.measure:
        print("--only and --measure are only for --round 3.", file=sys.stderr)
        return 1
    if args.bootstrap is None:
        args.bootstrap = 2000
    if args.permutation_test and args.model != "baseline":
        print("--permutation-test is only for --model baseline.", file=sys.stderr)
        return 1
    if args.model == "duration":
        return train_duration(args)
    if args.model == "baseline":
        return train_baseline(args)
    return train_cnn_model(args)  # cnn or cnn-aug: argparse only accepts TRAIN_MODELS


def cmd_interpretability(args: argparse.Namespace) -> int:
    from sonar_pipeline.interpret import run_interpretability

    report = run_interpretability(
        args.processed_dir, args.models_dir, args.results_dir, args.figures_dir
    )

    def top(entries, n=5):
        best = sorted(entries, key=lambda b: -b["value"])[:n]
        return ", ".join(
            f"{b['lo_hz']:g}-{b['hi_hz']:g} Hz ({b['value']:.3f})" for b in best
        )

    print("Interpretability on val (frozen models; test not used):")
    print(f"  baseline, top LOFAR bands: {top(report['baseline']['lofar'])}")
    print(f"  baseline, top DEMON bands: {top(report['baseline']['demon'], 3)}")
    for name, entry in report["cnn"].items():
        print(f"  {name}, top LOFAR occlusion: {top(entry['occlusion_lofar'])}")
    era = report["era_check"]
    print(f"  era lines ({era['era_line_rule']}): {era['era_lines_hz'] or 'none'}")
    for name, m in era["models"].items():
        print(
            f"  {name}: Spearman with the era difference {m['spearman_rho']:+.2f} "
            f"(p {m['spearman_p']:.3f}), top-5 overlap {m['top_overlap']}/5"
        )
    print(f"Written to {args.results_dir / 'interpretability.json'}")
    for path in report.get("figures", []):
        print(f"Figure: {path}")
    return 0


def interpretability_round3(args: argparse.Namespace) -> int:
    from sonar_pipeline.cv import Round3
    from sonar_pipeline.interpret_cv import (
        load_round3_summary,
        run_round3_interpretability,
    )

    summary_path = args.results_dir / "round3.json"
    engine = Round3(
        args.processed_dir,
        args.splits,
        args.models_dir / "round3",
        args.outputs_dir,
        device_name=args.device,
    )
    output = args.results_dir / "round3_interpretability.json"
    report = run_round3_interpretability(
        engine,
        load_round3_summary(summary_path),
        output,
        log=lambda line: print(line, flush=True),
    )
    print("Interpretability on val of the 25 partitions (test not used):")
    for name, entry in report["models"].items():
        top = ", ".join(
            f"{b['lo_hz']:g}-{b['hi_hz']:g}" for b in entry["top_lofar_bands"]
        )
        print(f"  {name:<16} top LOFAR bands (Hz): {top}")
    for name, m in report["era_check"]["models"].items():
        print(
            f"  {name}: Spearman with the era difference {m['spearman_rho']:+.2f}, "
            f"top-5 overlap {m['top_overlap']}/5"
        )
    print(f"Written to {output}")
    return 0


def details_round3(args: argparse.Namespace) -> int:
    import numpy as np

    from sonar_pipeline.cv import Round3
    from sonar_pipeline.interpret_cv import load_round3_summary
    from sonar_pipeline.round3_details import collect_details

    engine = Round3(
        args.processed_dir,
        args.splits,
        args.models_dir / "round3",
        args.outputs_dir,
        device_name=args.device,
    )
    output = args.results_dir / "round3_details.json"
    report = collect_details(
        engine,
        load_round3_summary(args.results_dir / "round3.json"),
        output,
        log=lambda line: print(line, flush=True),
    )
    print("Round 3 details (no training; test not rescored):")
    for name, entry in report["models"].items():
        gap = entry["generalisation"]
        train = np.mean([g["train_f1_macro"] for g in gap])
        test = np.mean([g["test_f1_macro"] for g in gap])
        print(
            f"  {name:<16} train F1 {train:.3f}  test F1 {test:.3f}  "
            f"ECE {entry['calibration']['ece']:.3f}"
        )
    print(f"Written to {output}")
    return 0


def check_round3_cmd(args: argparse.Namespace) -> int:
    from sonar_pipeline.check_round3 import check_round3
    from sonar_pipeline.interpret_cv import load_round3_summary

    result = check_round3(
        args.processed_dir,
        args.splits,
        args.models_dir / "round3" / "representative",
        load_round3_summary(args.results_dir / "round3.json"),
    )
    print(
        "Round 3 quick check (representative models, partition rep0_fold0, seed 0, CPU):"
    )
    for name, entry in result["models"].items():
        print(
            f"  {name:<16} test F1 {entry['test_f1_macro']:.4f}  reference "
            f"{entry['reference']:.4f}  difference {entry['difference']:+.4f}"
        )
    if result["problems"]:
        print("Does NOT reproduce the round 3 results:")
        for problem in result["problems"]:
            print(f"  {problem}")
        return 1
    print(
        f"Reproduces the round 3 results: same weights, every test F1 within "
        f"±{result['tolerance']:g}."
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if (args.check or args.details) and args.round != 3:
        # Without --round 3 the command would score the test split of the earlier evaluation.
        print("--check and --details need --round 3.", file=sys.stderr)
        return 1
    if args.results_dir is None:
        args.results_dir = Path(
            "docs/results" if args.round == 3 else "outputs/results"
        )
    if args.round == 3:
        if args.check:
            return check_round3_cmd(args)
        if args.details:
            return details_round3(args)
        if not args.interpretability:
            print(
                "--round 3 is only for --check, --interpretability or --details (test is "
                "scored by train --round 3).",
                file=sys.stderr,
            )
            return 1
        return interpretability_round3(args)
    if args.interpretability:
        return cmd_interpretability(args)
    if args.refresh:
        from sonar_pipeline.evaluate import refresh_metrics

        metrics = refresh_metrics(args.results_dir)
        sections = [
            k for k in ("dt005", "permutation_test", "interpretability") if metrics[k]
        ]
        print(
            f"metrics.json refreshed without scoring test: {', '.join(sections) or 'none'}"
        )
        return 0
    if args.plot_only:
        from sonar_pipeline.evaluate import plot_from_metrics

        for path in plot_from_metrics(args.results_dir, args.figures_dir):
            print(f"Figure: {path}")
        return 0
    from sonar_pipeline.evaluate import evaluate_models

    result = evaluate_models(
        args.processed_dir,
        args.models_dir,
        args.results_dir,
        args.figures_dir,
        args.device,
        args.bootstrap,
    )
    print("Test (scored once with the frozen, versioned models):")
    for name, entry in result["models"].items():
        test = entry["test"]
        line = f"  {name:<9}"
        for level in ("segment", "recording"):
            s = test[level]
            lo, hi = s["ci95"]["f1_macro"]
            line += f"  {level} F1 {s['f1_macro']:.3f} ({lo:.2f}-{hi:.2f})"
        print(line)
        eras = test["within_era"]
        print(
            "            within era (segment): "
            + "  ".join(
                f"{era} {eras[era]['classes'][0]} vs {eras[era]['classes'][1]} "
                f"{eras[era]['segment']['f1_macro']:.3f}"
                for era in eras
                if eras[era]["segment"] is not None
            )
        )
    print(f"Written to {args.results_dir / 'metrics.json'}")
    for path in result.get("figures", []):
        print(f"Figure: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from sonar_pipeline.report import run_report

    path = run_report(
        args.results_dir,
        args.figures_dir,
        args.manifests_dir,
        args.output,
        args.collect_figures,
    )
    print(f"Report written to {path} (from versioned files; test not scored)")
    return 0


HANDLERS = {
    "qc": cmd_qc,
    "report": cmd_report,
    "features": cmd_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
