#!/usr/bin/env python3
"""Audit completed YouTube stance runs without training or changing the ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


LABELS = ("찬성", "반대")
METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "precision_support",
    "recall_support",
    "f1_support",
    "precision_oppose",
    "recall_oppose",
    "f1_oppose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify train-only oversampling, split leakage, saved metrics, "
            "per-class F1/confusion matrices, and multi-seed stability"
        )
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Only include completed runs whose experiment_name exactly matches",
    )
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--max-test-macro-f1-std", type=float, default=0.05)
    parser.add_argument("--max-test-macro-f1-range", type=float, default=0.10)
    parser.add_argument("--metric-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--require-multiseed",
        action="store_true",
        help="Return failure when fewer than --min-seeds completed seeds exist",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def split_digest(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "comment_id",
            "video_id",
            "comment_hash",
            "leakage_group",
            "stance_label",
            "split",
        )
        if column in frame.columns
    ]
    canonical = frame[columns].sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Prediction rows and labels must be non-empty and aligned")
    unknown = (set(actual) | set(predicted)) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {sorted(unknown)}")

    result: dict[str, Any] = {
        "accuracy": sum(a == p for a, p in zip(actual, predicted, strict=True))
        / len(actual)
    }
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    suffixes = {"찬성": "support", "반대": "oppose"}
    matrix = [[0, 0], [0, 0]]

    for actual_label, predicted_label in zip(actual, predicted, strict=True):
        matrix[LABELS.index(actual_label)][LABELS.index(predicted_label)] += 1

    for label in LABELS:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        suffix = suffixes[label]
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    result["macro_precision"] = sum(precisions) / len(precisions)
    result["macro_recall"] = sum(recalls) / len(recalls)
    result["macro_f1"] = sum(f1_scores) / len(f1_scores)
    result["confusion_matrix_actual_rows_predicted_columns"] = matrix
    return result


def compare_metrics(
    recomputed: dict[str, Any], recorded: dict[str, Any], tolerance: float
) -> list[str]:
    errors: list[str] = []
    for key in METRIC_KEYS:
        if key not in recorded:
            errors.append(f"recorded metrics missing {key}")
            continue
        if not math.isclose(
            float(recomputed[key]), float(recorded[key]), rel_tol=0.0, abs_tol=tolerance
        ):
            errors.append(
                f"{key} mismatch: recomputed={recomputed[key]:.12f}, "
                f"recorded={float(recorded[key]):.12f}"
            )
    matrix_key = "confusion_matrix_actual_rows_predicted_columns"
    if recomputed[matrix_key] != recorded.get(matrix_key):
        errors.append(
            f"confusion matrix mismatch: recomputed={recomputed[matrix_key]}, "
            f"recorded={recorded.get(matrix_key)}"
        )
    return errors


def check_cross_split(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"status": "fail", "cross_split_values": None, "error": f"missing {column}"}
    counts = frame.groupby(column, dropna=False)["split"].nunique()
    leaked = counts[counts > 1]
    return {
        "status": "pass" if leaked.empty else "fail",
        "cross_split_values": int(len(leaked)),
        "examples": [str(value) for value in leaked.index[:10]],
    }


def audit_sampling(
    run_dir: Path, manifest: pd.DataFrame, summary: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    sampling_path = run_dir / "training_sampling_manifest.csv"
    sampling_summary = summary.get("training_sampling", {})
    if not sampling_summary.get("enabled"):
        return {"status": "fail", "errors": ["oversampling is not enabled"]}
    try:
        sampling = read_csv(sampling_path)
    except Exception as error:
        return {"status": "fail", "errors": [f"cannot read sampling manifest: {error}"]}

    required = {
        "training_sample_index",
        "oversampled_copy",
        "comment_id",
        "video_id",
        "stance_label",
    }
    missing = sorted(required - set(sampling.columns))
    if missing:
        return {"status": "fail", "errors": [f"missing columns: {missing}"]}

    train = manifest.loc[manifest["split"].eq("train")].copy()
    dev_test_ids = set(manifest.loc[manifest["split"].isin(("dev", "test")), "comment_id"])
    sample_ids = set(sampling["comment_id"])
    original_mask = ~as_bool(sampling["oversampled_copy"])
    copy_mask = ~original_mask
    original_ids = sampling.loc[original_mask, "comment_id"]

    split_by_comment = manifest.set_index("comment_id")["split"].to_dict()
    unknown_ids = sorted(sample_ids - set(split_by_comment))
    non_train_ids = sorted(
        comment_id
        for comment_id in sample_ids
        if split_by_comment.get(comment_id) not in (None, "train")
    )
    if unknown_ids:
        errors.append(f"sampling contains {len(unknown_ids)} unknown comment IDs")
    if non_train_ids or sample_ids & dev_test_ids:
        errors.append(
            f"sampling contains {len(set(non_train_ids) | (sample_ids & dev_test_ids))} dev/test IDs"
        )
    if set(original_ids) != set(train["comment_id"]) or len(original_ids) != len(train):
        errors.append("non-copy sampling rows do not exactly match the original train rows")
    if original_ids.duplicated().any():
        errors.append("original train rows are duplicated before oversampling copies")
    if not sampling.loc[copy_mask, "stance_label"].eq("찬성").all():
        errors.append("one or more oversampled copies are not 찬성")

    before_counts = Counter(train["stance_label"])
    after_counts = Counter(sampling["stance_label"])
    expected_before = {str(k): int(v) for k, v in sampling_summary.get("before_label_counts", {}).items()}
    expected_after = {str(k): int(v) for k, v in sampling_summary.get("after_label_counts", {}).items()}
    if dict(before_counts) != expected_before:
        errors.append(f"before label counts mismatch: {dict(before_counts)} != {expected_before}")
    if dict(after_counts) != expected_after:
        errors.append(f"after label counts mismatch: {dict(after_counts)} != {expected_after}")
    if len(sampling) != int(sampling_summary.get("after_rows", -1)):
        errors.append("after row count does not match run_summary.json")
    added_copies = int(copy_mask.sum())
    if added_copies != int(sampling_summary.get("added_support_copies", -1)):
        errors.append("added support copy count does not match run_summary.json")

    actual_ratio = after_counts["찬성"] / after_counts["반대"]
    recorded_ratio = float(sampling_summary.get("actual_support_to_oppose_ratio", -1))
    if not math.isclose(actual_ratio, recorded_ratio, rel_tol=0.0, abs_tol=1e-12):
        errors.append("actual support:oppose ratio does not match run_summary.json")
    if sampling_summary.get("scope") != "train_only" or sampling_summary.get(
        "dev_test_oversampled"
    ) is not False:
        errors.append("run_summary.json does not declare train-only oversampling")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "original_train_rows": int(len(train)),
        "training_rows_after_oversampling": int(len(sampling)),
        "added_support_copies": added_copies,
        "before_label_counts": dict(before_counts),
        "after_label_counts": dict(after_counts),
        "actual_support_to_oppose_ratio": actual_ratio,
        "dev_test_ids_in_sampling": int(len(sample_ids & dev_test_ids)),
    }


def audit_predictions(
    run_dir: Path,
    manifest: pd.DataFrame,
    summary: dict[str, Any],
    split: str,
    tolerance: float,
) -> dict[str, Any]:
    errors: list[str] = []
    path = run_dir / f"{split}_predictions.csv"
    try:
        predictions = read_csv(path)
    except Exception as error:
        return {"status": "fail", "errors": [f"cannot read predictions: {error}"]}
    required = {"comment_id", "stance_label", "predicted_label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        return {"status": "fail", "errors": [f"missing columns: {missing}"]}

    expected_ids = set(manifest.loc[manifest["split"].eq(split), "comment_id"])
    actual_ids = set(predictions["comment_id"])
    if predictions["comment_id"].duplicated().any():
        errors.append("prediction comment IDs are duplicated")
    if actual_ids != expected_ids or len(predictions) != len(expected_ids):
        errors.append(
            f"prediction population mismatch: rows={len(predictions)}, expected={len(expected_ids)}, "
            f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
        )

    recomputed = classification_metrics(
        predictions["stance_label"].tolist(), predictions["predicted_label"].tolist()
    )
    errors.extend(compare_metrics(recomputed, summary.get(f"{split}_metrics", {}), tolerance))
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "rows": int(len(predictions)),
        "recomputed_metrics": recomputed,
    }


def audit_run(run_dir: Path, summary: dict[str, Any], tolerance: float) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest = read_csv(run_dir / "split_manifest.csv")
    except Exception as error:
        return {
            "run_id": summary.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "status": "fail",
            "errors": [f"cannot read split manifest: {error}"],
        }

    required = {
        "comment_id",
        "video_id",
        "comment_hash",
        "leakage_group",
        "stance_label",
        "split",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        errors.append(f"split manifest missing columns: {missing}")
    unexpected_splits = sorted(set(manifest.get("split", pd.Series(dtype=str))) - {"train", "dev", "test"})
    if unexpected_splits:
        errors.append(f"unexpected split names: {unexpected_splits}")
    if manifest.get("comment_id", pd.Series(dtype=str)).duplicated().any():
        errors.append("split manifest contains duplicate comment IDs")

    leakage_checks = {
        column: check_cross_split(manifest, column)
        for column in ("comment_id", "video_id", "comment_hash", "leakage_group")
    }
    errors.extend(
        f"{column} crosses splits ({check.get('cross_split_values')})"
        for column, check in leakage_checks.items()
        if check["status"] != "pass"
    )

    text_check: dict[str, Any]
    data_path = Path(str(summary.get("arguments", {}).get("data", "")))
    try:
        source = read_csv(data_path)
        if not {"comment_id", "comment_text"}.issubset(source.columns):
            raise ValueError("source CSV lacks comment_id/comment_text")
        source_text = source[["comment_id", "comment_text"]].drop_duplicates("comment_id")
        joined = manifest[["comment_id", "split"]].merge(
            source_text, on="comment_id", how="left", validate="one_to_one"
        )
        missing_text = int(
            (joined["comment_text"].isna() | joined["comment_text"].eq("")).sum()
        )
        joined["comment_text"] = joined["comment_text"].fillna("")
        joined["normalized_comment_text"] = joined["comment_text"].map(normalized_text)
        text_counts = joined.groupby("normalized_comment_text")["split"].nunique()
        leaked_text = text_counts[(text_counts > 1) & text_counts.index.to_series().ne("")]
        text_check = {
            "status": "pass" if missing_text == 0 and leaked_text.empty else "fail",
            "source_csv": str(data_path),
            "missing_comment_text_rows": missing_text,
            "normalized_texts_crossing_splits": int(len(leaked_text)),
            "examples": [str(value)[:160] for value in leaked_text.index[:10]],
        }
        if text_check["status"] != "pass":
            errors.append(
                "normalized comment text leakage or missing source text detected: "
                f"missing={missing_text}, crossing={len(leaked_text)}"
            )
    except Exception as error:
        text_check = {"status": "fail", "error": str(error), "source_csv": str(data_path)}
        errors.append(f"cannot independently verify normalized comment text: {error}")

    expected_split_summary = summary.get("split_summary", {})
    split_population: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        rows = int(manifest["split"].eq(split).sum())
        recorded_rows = int(expected_split_summary.get(split, {}).get("rows", -1))
        status = "pass" if rows == recorded_rows else "fail"
        split_population[split] = {
            "status": status,
            "manifest_rows": rows,
            "recorded_rows": recorded_rows,
        }
        if status == "fail":
            errors.append(f"{split} row count mismatch: {rows} != {recorded_rows}")

    sampling = audit_sampling(run_dir, manifest, summary)
    dev = audit_predictions(run_dir, manifest, summary, "dev", tolerance)
    test = audit_predictions(run_dir, manifest, summary, "test", tolerance)
    for name, check in (("sampling", sampling), ("dev metrics", dev), ("test metrics", test)):
        if check["status"] != "pass":
            errors.extend(f"{name}: {error}" for error in check.get("errors", []))

    arguments = summary.get("arguments", {})
    split_seed = arguments.get("split_seed")
    if split_seed is None:
        split_seed = summary.get("data_audit", {}).get("split_seed", arguments.get("seed"))
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "experiment_name": summary.get("experiment_name"),
        "training_seed": arguments.get("seed"),
        "split_seed": split_seed,
        "split_digest": split_digest(manifest),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "split_population": split_population,
        "oversampling_train_only": sampling,
        "cross_split_leakage": leakage_checks,
        "normalized_comment_text_leakage": text_check,
        "dev": dev,
        "test": test,
    }


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def audit_multiseed(
    runs: list[dict[str, Any]], min_seeds: int, max_std: float, max_range: float
) -> dict[str, Any]:
    passed = [run for run in runs if run["status"] == "pass"]
    seeds = [run.get("training_seed") for run in passed]
    unique_seeds = sorted({int(seed) for seed in seeds if seed is not None})
    selected_by_seed: dict[int, dict[str, Any]] = {}
    for run in passed:
        if run.get("training_seed") is not None:
            selected_by_seed[int(run["training_seed"])] = run
    selected_runs = [selected_by_seed[seed] for seed in unique_seeds]
    split_digests = sorted({run["split_digest"] for run in passed})
    split_seeds = sorted({int(run["split_seed"]) for run in passed if run.get("split_seed") is not None})
    errors: list[str] = []
    if len(unique_seeds) < min_seeds:
        errors.append(f"completed unique seeds={len(unique_seeds)}; required={min_seeds}")
    if len(split_digests) > 1:
        errors.append("split manifests differ across seeds")
    if len(split_seeds) > 1:
        errors.append(f"split seeds differ across runs: {split_seeds}")

    metrics: dict[str, Any] = {}
    for key in ("accuracy", "macro_f1", "f1_support", "f1_oppose"):
        values = [
            float(run["test"]["recomputed_metrics"][key]) for run in selected_runs
        ]
        if values:
            metrics[key] = summarize_metric(values)
    macro = metrics.get("macro_f1")
    if macro and macro["sample_std"] > max_std:
        errors.append(
            f"test macro F1 sample std={macro['sample_std']:.6f} exceeds {max_std:.6f}"
        )
    if macro and macro["range"] > max_range:
        errors.append(f"test macro F1 range={macro['range']:.6f} exceeds {max_range:.6f}")

    insufficient_only = bool(errors) and all(
        error.startswith("completed unique seeds=") for error in errors
    )
    status = "pass" if not errors else ("pending" if insufficient_only else "fail")
    return {
        "status": status,
        "errors": errors,
        "unique_training_seeds": unique_seeds,
        "selected_run_ids_by_seed": {
            str(seed): selected_by_seed[seed]["run_id"] for seed in unique_seeds
        },
        "split_seeds": split_seeds,
        "identical_split_manifests": len(split_digests) <= 1,
        "thresholds": {
            "minimum_completed_seeds": min_seeds,
            "maximum_test_macro_f1_sample_std": max_std,
            "maximum_test_macro_f1_range": max_range,
        },
        "test_metrics": metrics,
    }


def matrix_markdown(matrix: list[list[int]]) -> list[str]:
    return [
        "| 실제 \\ 예측 | 찬성 | 반대 |",
        "|---|---:|---:|",
        f"| 찬성 | {matrix[0][0]} | {matrix[0][1]} |",
        f"| 반대 | {matrix[1][0]} | {matrix[1][1]} |",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# YouTube 입장 분류 실험 감사 보고서",
        "",
        f"- 전체 상태: **{report['overall_status']}**",
        f"- 생성 시각(KST): `{report['generated_at_kst']}`",
        f"- 완료 run 수: {len(report['runs'])}",
        "- 판정 방식: 저장된 예측 파일로 지표를 독립 재계산하고, 분할표와 샘플링표의 모집단을 대조했습니다.",
        "",
        "## 핵심 점검",
        "",
        "| run | 학습 seed | split seed | train-only 오버샘플링 | 누수 없음 | dev 지표 | test 지표 |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for run in report["runs"]:
        no_leakage = all(
            check["status"] == "pass" for check in run.get("cross_split_leakage", {}).values()
        ) and run.get("normalized_comment_text_leakage", {}).get("status") == "pass"
        lines.append(
            f"| {run['run_id']} | {run.get('training_seed', '-')} | {run.get('split_seed', '-')} | "
            f"{run.get('oversampling_train_only', {}).get('status', 'fail')} | "
            f"{'pass' if no_leakage else 'fail'} | {run.get('dev', {}).get('status', 'fail')} | "
            f"{run.get('test', {}).get('status', 'fail')} |"
        )

    for run in report["runs"]:
        lines.extend(["", f"## {run['run_id']}", "", f"- 상태: **{run['status']}**"])
        if run.get("errors"):
            lines.append("- 오류: " + "; ".join(run["errors"]))
        sampling = run.get("oversampling_train_only", {})
        if sampling.get("status"):
            lines.append(
                "- 오버샘플링: "
                f"{sampling['status']} · 원본 train {sampling.get('original_train_rows', '-')}행 → "
                f"{sampling.get('training_rows_after_oversampling', '-')}행 · "
                f"찬성 복제 {sampling.get('added_support_copies', '-')}행 · "
                f"dev/test 유입 {sampling.get('dev_test_ids_in_sampling', '-')}행"
            )
        for split in ("dev", "test"):
            check = run.get(split, {})
            metrics = check.get("recomputed_metrics")
            if not metrics:
                continue
            lines.extend(
                [
                    "",
                    f"### {split} 독립 재계산",
                    "",
                    "| Accuracy | Macro Precision | Macro Recall | Macro F1 | 찬성 F1 | 반대 F1 |",
                    "|---:|---:|---:|---:|---:|---:|",
                    (
                        f"| {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | "
                        f"{metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} | "
                        f"{metrics['f1_support']:.4f} | {metrics['f1_oppose']:.4f} |"
                    ),
                    "",
                    *matrix_markdown(metrics["confusion_matrix_actual_rows_predicted_columns"]),
                ]
            )

    multi = report["multiseed"]
    lines.extend(
        [
            "",
            "## 다중 seed 안정성",
            "",
            f"- 상태: **{multi['status']}**",
            f"- 학습 seed: `{multi['unique_training_seeds']}`",
            f"- split seed: `{multi['split_seeds']}`",
            f"- 분할표 동일: `{multi['identical_split_manifests']}`",
        ]
    )
    if multi["errors"]:
        lines.append("- 판정 사유: " + "; ".join(multi["errors"]))
    if multi["test_metrics"]:
        lines.extend(
            [
                "",
                "| test 지표 | 평균 | 표본 표준편차 | 최솟값 | 최댓값 | 범위 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        display = {
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "f1_support": "찬성 F1",
            "f1_oppose": "반대 F1",
        }
        for key, values in multi["test_metrics"].items():
            lines.append(
                f"| {display[key]} | {values['mean']:.4f} | {values['sample_std']:.4f} | "
                f"{values['min']:.4f} | {values['max']:.4f} | {values['range']:.4f} |"
            )
    lines.extend(
        [
            "",
            "> 다중 seed가 pending이면 단일 seed의 무결성은 확인됐지만 seed 안정성은 아직 입증되지 않은 상태입니다.",
            "> 이 평가는 LLM silver 라벨과의 일치도이며 사람 gold 성능으로 해석하면 안 됩니다.",
            "",
        ]
    )
    return "\n".join(lines)


def discover_runs(root: Path, experiment_name: str | None) -> list[tuple[Path, dict[str, Any]]]:
    candidates = [root / "run_summary.json"] if (root / "run_summary.json").is_file() else sorted(root.glob("*/run_summary.json"))
    runs: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in candidates:
        with summary_path.open(encoding="utf-8") as source:
            summary = json.load(source)
        if summary.get("status") != "completed":
            continue
        if experiment_name and summary.get("experiment_name") != experiment_name:
            continue
        runs.append((summary_path.parent, summary))
    return runs


def main() -> None:
    args = parse_args()
    if args.min_seeds < 1:
        raise ValueError("--min-seeds must be at least 1")
    discovered = discover_runs(args.runs_root, args.experiment_name)
    if not discovered:
        raise FileNotFoundError("No matching completed run_summary.json files found")

    runs = [audit_run(run_dir, summary, args.metric_tolerance) for run_dir, summary in discovered]
    multiseed = audit_multiseed(
        runs,
        args.min_seeds,
        args.max_test_macro_f1_std,
        args.max_test_macro_f1_range,
    )
    run_failures = any(run["status"] != "pass" for run in runs)
    multiseed_failure = multiseed["status"] == "fail" or (
        args.require_multiseed and multiseed["status"] != "pass"
    )
    overall_status = (
        "failed"
        if run_failures or multiseed_failure
        else ("pending_multiseed" if multiseed["status"] == "pending" else "ready")
    )
    report = {
        "overall_status": overall_status,
        "generated_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "runs_root": str(args.runs_root),
        "experiment_name_filter": args.experiment_name,
        "require_multiseed": args.require_multiseed,
        "runs": runs,
        "multiseed": multiseed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit_report.json"
    markdown_path = args.output_dir / "audit_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": overall_status, "json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False, indent=2))
    if overall_status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
