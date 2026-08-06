#!/usr/bin/env python3
"""Run grouped 5-fold encoder training with held-out threshold calibration.

The group-safe 80/10/10 split has three distinct roles:

* train 80%: five-fold cross-validation and model fitting
* dev 10%: ensemble decision-threshold calibration only
* test 10%: one final evaluation after thresholds are frozen

Every split and fold assignment is made at ``leakage_group`` level.  A
leakage group connects videos that share a duplicated comment, so neither a
video nor an exact comment can cross the train/dev/test or fold boundaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)

from train_youtube_stance_roberta import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    SOURCE_TO_STANCE,
    SplitRatios,
    TokenizedCommentDataset,
    WeightedLossTrainer,
    add_leakage_groups,
    calculate_class_weights,
    classification_metrics,
    compute_metrics,
    confusion_matrix,
    json_ready,
    load_excluded_video_ids,
    load_silver_data,
    make_group_summary,
    make_grouped_splits,
    make_run_id,
    make_training_arguments,
    now_kst,
    runtime_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train five encoder folds on the 80% training pool, "
            "calibrate thresholds on dev 10%, and evaluate test 10% once"
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--experiment-name", default="youtube-stance-klue-roberta-base-cv5-e3"
    )
    parser.add_argument("--experiment-note", default="E02")
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--split-search-trials", type=int, default=5000)
    parser.add_argument("--fold-search-trials", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--text-mode", choices=("comment_context", "comment"), default="comment_context"
    )
    parser.add_argument(
        "--context-mode",
        choices=("metadata", "law_original"),
        default="metadata",
        help=(
            "metadata keeps the legacy context; law_original uses law name, "
            "expression type, and verbatim excerpts selected from law JSON files"
        ),
    )
    parser.add_argument(
        "--law-body-dir",
        type=Path,
        default=None,
        help="Directory containing one <law_id>.json source file per law",
    )
    parser.add_argument("--law-original-top-sections", type=int, default=3)
    parser.add_argument("--law-original-max-chars", type=int, default=600)
    parser.add_argument("--exclude-video-ids", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "sqrt_balanced"),
        default="sqrt_balanced",
    )
    parser.add_argument("--include-needs-review", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Unsupported for CV runs; retained only for TrainingArguments compatibility",
    )
    return parser.parse_args()


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _content_strings(value: Any, parent_key: str = "") -> list[str]:
    """Return only verbatim *content* fields from a law-service payload."""

    strings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            strings.extend(_content_strings(child, str(key)))
    elif isinstance(value, list):
        for child in value:
            strings.extend(_content_strings(child, parent_key))
    elif isinstance(value, str) and "내용" in parent_key:
        text = _normalized_text(value)
        if text:
            strings.append(text)
    return strings


def _deduplicated_join(strings: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for string in strings:
        text = _normalized_text(string)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return " ".join(ordered)


def _law_candidates(payload: dict[str, Any]) -> list[tuple[str, str]]:
    law = payload.get("response", {}).get("법령", {})
    if not isinstance(law, dict):
        raise ValueError("Law JSON does not contain response.법령")

    candidates: list[tuple[str, str]] = []
    articles = law.get("조문", {}).get("조문단위", [])
    if isinstance(articles, dict):
        articles = [articles]
    if isinstance(articles, list):
        for article in articles:
            text = _deduplicated_join(_content_strings(article))
            if text:
                candidates.append(("조문", text))

    for source_type, key in (("개정 이유", "제개정이유"), ("개정문", "개정문")):
        text = _deduplicated_join(_content_strings(law.get(key)))
        if text:
            # These fields can contain a full multi-page document. Keeping
            # paragraph-sized original chunks lets relevance selection work
            # without rewriting or summarizing the source language.
            chunks = [
                text[index : index + 800]
                for index in range(0, len(text), 800)
                if text[index : index + 800]
            ]
            candidates.extend((source_type, chunk) for chunk in chunks)

    deduplicated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source_type, text in candidates:
        if text not in seen:
            seen.add(text)
            deduplicated.append((source_type, text))
    return deduplicated


def _character_ngrams(text: str, size: int = 2) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z가-힣]", "", text.lower())
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _select_original_sections(
    comment: str,
    candidates: list[tuple[str, str]],
    top_sections: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    query_ngrams = _character_ngrams(comment)
    source_priority = {"조문": 2, "개정 이유": 1, "개정문": 0}

    def rank(candidate: tuple[str, str]) -> tuple[float, int, int]:
        source_type, text = candidate
        candidate_ngrams = _character_ngrams(text)
        overlap = len(query_ngrams & candidate_ngrams)
        score = overlap / max(1.0, math.sqrt(len(candidate_ngrams)))
        return score, source_priority.get(source_type, -1), -len(text)

    selected = sorted(candidates, key=rank, reverse=True)[:top_sections]
    remaining = max_chars
    excerpts: list[str] = []
    source_types: list[str] = []
    for source_type, original_text in selected:
        delimiter_chars = 1 if excerpts else 0
        available = remaining - delimiter_chars
        if available <= 0:
            break
        # Slice the original string directly. No summary, paraphrase, or
        # generated connective text is introduced into this feature value.
        excerpt = original_text[:available]
        if excerpt:
            excerpts.append(excerpt)
            source_types.append(source_type)
            remaining -= delimiter_chars + len(excerpt)
    return "\n".join(excerpts), source_types


def apply_law_original_context(
    frame: pd.DataFrame,
    law_body_dir: Path,
    top_sections: int,
    max_chars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"law_id", "law_name", "expression_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"law_original context is missing CSV columns: {missing}")
    if not law_body_dir.is_dir():
        raise FileNotFoundError(f"Law body directory does not exist: {law_body_dir}")
    if top_sections < 1 or max_chars < 1:
        raise ValueError("Law original section/character limits must be positive")

    result = frame.copy()
    for column in required:
        result[column] = result[column].astype(str).str.strip()
    if result[list(required)].eq("").any(axis=None):
        raise ValueError("law_original context requires non-empty law_id/law_name/expression_label")
    invalid_expression = sorted(
        set(result["expression_label"]) - {"일반", "반어"}
    )
    if invalid_expression:
        raise ValueError(f"Unexpected expression labels: {invalid_expression}")

    candidate_by_law: dict[str, list[tuple[str, str]]] = {}
    law_file_by_id: dict[str, Path] = {}
    file_hashes: dict[str, str] = {}
    for law_id in sorted(result["law_id"].unique()):
        normalized_id = law_id.zfill(6) if law_id.isdigit() else law_id
        path = law_body_dir / f"{normalized_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing law source JSON for law_id={law_id}: {path}")
        with path.open("r", encoding="utf-8-sig") as source:
            payload = json.load(source)
        candidates = _law_candidates(payload)
        if not candidates:
            raise ValueError(f"No original provisions/amendment text found in {path}")
        candidate_by_law[law_id] = candidates
        law_file_by_id[law_id] = path
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    original_texts: list[str] = []
    source_type_values: list[str] = []
    for row in result.itertuples(index=False):
        original_text, source_types = _select_original_sections(
            str(row.comment_text),
            candidate_by_law[str(row.law_id)],
            top_sections,
            max_chars,
        )
        if not original_text:
            raise ValueError(f"Could not select law text for comment_id={row.comment_id}")
        original_texts.append(original_text)
        source_type_values.append(",".join(source_types))

    result["law_original_text"] = original_texts
    result["law_original_source_types"] = source_type_values
    result["context_text"] = result.apply(
        lambda row: (
            f"법률명: {row['law_name']} | 표현 유형: {row['expression_label']} | "
            f"관련 조문·개정 이유 원문: {row['law_original_text']}"
        ),
        axis=1,
    )

    combined_hash = hashlib.sha256()
    for filename, digest in sorted(file_hashes.items()):
        combined_hash.update(f"{filename}:{digest}\n".encode("utf-8"))
    char_counts = result["law_original_text"].str.len()
    audit = {
        "mode": "law_original",
        "feature_fields": [
            "comment_text",
            "law_name",
            "expression_label",
            "law_original_text",
        ],
        "target_excluded_from_features": True,
        "video_title_excluded": True,
        "source_policy": "verbatim_original_excerpt_no_summary_or_paraphrase",
        "law_body_dir": str(law_body_dir),
        "law_file_count": len(file_hashes),
        "law_files_combined_sha256": combined_hash.hexdigest(),
        "top_sections": top_sections,
        "max_original_chars": max_chars,
        "original_characters": {
            "min": int(char_counts.min()),
            "median": float(char_counts.median()),
            "max": int(char_counts.max()),
        },
    }
    return result, audit


def fold_assignment_score(
    group_summary: pd.DataFrame,
    assignment: dict[str, int],
    num_folds: int,
) -> float | None:
    total_rows = int(group_summary["rows"].sum())
    label_columns = [f"label_{label_id}" for label_id in range(len(LABELS))]
    total_class_share = group_summary[label_columns].sum().to_numpy() / total_rows
    assigned_fold = pd.Series(assignment, dtype=int)
    score = 0.0
    for fold_index in range(num_folds):
        selected_groups = assigned_fold.index[assigned_fold.eq(fold_index)]
        if selected_groups.empty:
            return None
        part = group_summary.loc[selected_groups]
        class_counts = part[label_columns].sum().to_numpy()
        if np.any(class_counts == 0):
            return None
        part_rows = int(part["rows"].sum())
        score += 5.0 * (part_rows / total_rows - 1.0 / num_folds) ** 2
        score += float(np.square(class_counts / part_rows - total_class_share).sum())
    return score


def assign_grouped_folds(
    train_frame: pd.DataFrame,
    num_folds: int,
    seed: int,
    trials: int,
) -> dict[str, int]:
    groups = sorted(train_frame["leakage_group"].unique())
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    if len(groups) < num_folds:
        raise ValueError(
            f"Training pool has {len(groups)} leakage groups, fewer than {num_folds} folds"
        )

    rng = random.Random(seed + 1_000_003)
    group_summary = make_group_summary(train_frame)
    best_assignment: dict[str, int] | None = None
    best_score = float("inf")
    for _ in range(trials):
        assignment = {group: rng.randrange(num_folds) for group in groups}
        score = fold_assignment_score(group_summary, assignment, num_folds)
        if score is not None and score < best_score:
            best_assignment = assignment
            best_score = score

    if best_assignment is None:
        raise RuntimeError(
            "Could not create group-safe folds containing every label. "
            "Increase --fold-search-trials or inspect label distribution by video."
        )
    return best_assignment


def summarize_partition(part: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(part)),
        "videos": int(part["video_id"].nunique()),
        "leakage_groups": int(part["leakage_group"].nunique()),
        "labels": {
            key: int(value) for key, value in part["stance_label"].value_counts().items()
        },
    }


def predict_probabilities(
    trainer: Trainer,
    dataset: TokenizedCommentDataset,
) -> np.ndarray:
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits_tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    return torch.softmax(logits_tensor, dim=-1).cpu().numpy()


def metrics_with_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = classification_metrics(labels, predictions)
    metrics["confusion_matrix_actual_rows_predicted_columns"] = confusion_matrix(
        labels, predictions
    )
    return metrics


def threshold_predictions(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Apply class-specific thresholds using normalized probability scores.

    Neutral is the fixed reference threshold (0.5). Support and opposition
    thresholds are selected on the held-out dev set.  Dividing each class
    probability by its threshold makes the decision rule explicit and keeps
    every row assigned to exactly one of the three classes.
    """

    if probabilities.shape[1] == 2:
        return np.where(probabilities[:, 0] >= thresholds[0], 0, 1)
    return np.argmax(probabilities / thresholds.reshape(1, -1), axis=1)


def tune_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not (0.0 < minimum <= maximum < 1.0):
        raise ValueError("Threshold bounds must satisfy 0 < min <= max < 1")
    if step <= 0:
        raise ValueError("--threshold-step must be greater than zero")

    candidates = np.arange(minimum, maximum + step / 2.0, step, dtype=np.float64)
    best_key: tuple[float, float, float] | None = None
    best_thresholds: np.ndarray | None = None
    best_metrics: dict[str, Any] | None = None

    if len(LABELS) == 2:
        for support_threshold in candidates:
            thresholds = np.asarray(
                [support_threshold, 1.0 - support_threshold], dtype=np.float64
            )
            predictions = threshold_predictions(probabilities, thresholds)
            metrics = classification_metrics(labels, predictions)
            key = (
                metrics["macro_f1"],
                metrics["accuracy"],
                -abs(support_threshold - 0.5),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = thresholds
                best_metrics = metrics
    else:
        # Neutral stays at 0.5 as the reference scale. The support/opposition
        # thresholds are exhaustively calibrated only on the held-out dev 10%.
        for support_threshold in candidates:
            for oppose_threshold in candidates:
                thresholds = np.asarray(
                    [support_threshold, oppose_threshold, 0.5], dtype=np.float64
                )
                predictions = threshold_predictions(probabilities, thresholds)
                metrics = classification_metrics(labels, predictions)
                key = (
                    metrics["macro_f1"],
                    metrics["accuracy"],
                    -abs(support_threshold - 0.5) - abs(oppose_threshold - 0.5),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = thresholds
                    best_metrics = metrics

    if best_thresholds is None or best_metrics is None:
        raise RuntimeError("Threshold calibration produced no candidate")
    search_summary = {
        "objective": "dev_macro_f1",
        "decision_rule": (
            "support_if_P(support)>=threshold_else_oppose"
            if len(LABELS) == 2
            else "argmax(probability / class_threshold)"
        ),
        "candidate_min": minimum,
        "candidate_max": maximum,
        "candidate_step": step,
        "candidate_count": int(
            len(candidates) if len(LABELS) == 2 else len(candidates) ** 2
        ),
        "selected": {
            ID2LABEL[label_id]: float(best_thresholds[label_id])
            for label_id in range(len(LABELS))
        },
        "selected_dev_macro_f1": float(best_metrics["macro_f1"]),
    }
    return best_thresholds, search_summary


def prediction_frame(
    dataset: TokenizedCommentDataset,
    probabilities: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "comment_id",
        "video_id",
        "comment_text",
        "sentiment_label",
        "stance_label",
    ]
    for optional_column in (
        "law_id",
        "law_name",
        "expression_label",
        "law_original_source_types",
        "law_original_text",
        "context_text",
    ):
        if optional_column in dataset.frame.columns:
            columns.append(optional_column)
    output = dataset.frame[columns].copy()
    actual_ids = dataset.frame["label_id"].to_numpy(dtype=np.int64)
    output["raw_predicted_label"] = [ID2LABEL[int(value)] for value in raw_predictions]
    output["calibrated_predicted_label"] = [
        ID2LABEL[int(value)] for value in calibrated_predictions
    ]
    output["raw_correct"] = actual_ids == raw_predictions
    output["calibrated_correct"] = actual_ids == calibrated_predictions
    for label_id, label_name in ID2LABEL.items():
        output[f"ensemble_prob_{label_name}"] = probabilities[:, label_id]
    return output


def aggregate_fold_metrics(
    fold_results: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    metric_keys = sorted(
        key
        for key, value in fold_results[0]["validation_metrics"].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in metric_keys:
        values = np.asarray(
            [result["validation_metrics"][key] for result in fold_results],
            dtype=np.float64,
        )
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def save_fold_metrics_csv(fold_results: list[dict[str, Any]], path: Path) -> None:
    records = []
    for result in fold_results:
        record = {
            "fold": result["fold"],
            "train_rows": result["train_rows"],
            "validation_rows": result["validation_rows"],
            "best_checkpoint": result["best_checkpoint"],
            "best_metric": result["best_metric"],
            "train_loss": result["train_metrics"].get("train_loss"),
            "train_runtime": result["train_metrics"].get("train_runtime"),
        }
        record.update(result["validation_metrics"])
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def validate_args(args: argparse.Namespace) -> None:
    SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio).validate()
    if args.split_search_trials < 1 or args.fold_search_trials < 1:
        raise ValueError("Split/fold search trials must be at least 1")
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.resume_from_checkpoint:
        raise ValueError("CV E02 does not support --resume-from-checkpoint")
    if args.context_mode == "law_original" and args.text_mode != "comment_context":
        raise ValueError("law_original context requires --text-mode comment_context")
    if args.context_mode == "law_original" and args.law_body_dir is None:
        raise ValueError("law_original context requires --law-body-dir")


def run_experiment(args: argparse.Namespace, started_at_kst: str) -> dict[str, Any]:
    validate_args(args)
    total_started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    excluded_video_ids = load_excluded_video_ids(args.exclude_video_ids)
    frame, data_audit = load_silver_data(
        args.data, args.include_needs_review, excluded_video_ids
    )
    if args.context_mode == "law_original":
        frame, feature_audit = apply_law_original_context(
            frame,
            args.law_body_dir,
            args.law_original_top_sections,
            args.law_original_max_chars,
        )
        data_audit["feature_context"] = feature_audit
    else:
        data_audit["feature_context"] = {
            "mode": "metadata",
            "target_excluded_from_features": True,
        }
    frame = add_leakage_groups(frame)
    frame = make_grouped_splits(
        frame,
        ratios=SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio),
        seed=args.seed,
        trials=args.split_search_trials,
    )

    train_mask = frame["split"].eq("train")
    fold_assignment = assign_grouped_folds(
        frame.loc[train_mask], args.num_folds, args.seed, args.fold_search_trials
    )
    frame["cv_fold"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[train_mask, "cv_fold"] = (
        frame.loc[train_mask, "leakage_group"].map(fold_assignment).astype("Int64")
    )

    manifest_columns = [
        "comment_id",
        "video_id",
        "comment_hash",
        "leakage_group",
        "sentiment_label",
        "stance_label",
        "label_id",
        "split",
        "cv_fold",
    ]
    frame[manifest_columns].to_csv(
        args.output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    if args.context_mode == "law_original":
        feature_manifest_columns = [
            "comment_id",
            "video_id",
            "law_id",
            "law_name",
            "expression_label",
            "law_original_source_types",
            "law_original_text",
            "context_text",
            "split",
            "cv_fold",
        ]
        frame[feature_manifest_columns].to_csv(
            args.output_dir / "feature_context_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
    split_summary = {
        split: summarize_partition(frame.loc[frame["split"].eq(split)])
        for split in ("train", "dev", "test")
    }

    model_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    return_token_type_ids = bool(getattr(model_config, "type_vocab_size", 1) > 1)
    args.return_token_type_ids = return_token_type_ids
    use_context = args.text_mode == "comment_context"
    calibration_dataset = TokenizedCommentDataset(
        frame.loc[frame["split"].eq("dev")].copy(),
        tokenizer,
        args.max_length,
        use_context,
        return_token_type_ids=return_token_type_ids,
    )
    test_dataset = TokenizedCommentDataset(
        frame.loc[frame["split"].eq("test")].copy(),
        tokenizer,
        args.max_length,
        use_context,
        return_token_type_ids=return_token_type_ids,
    )
    calibration_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    fold_results: list[dict[str, Any]] = []

    for fold_index in range(args.num_folds):
        fold_number = fold_index + 1
        fold_dir = args.output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_train = frame.loc[train_mask & frame["cv_fold"].ne(fold_index)].copy()
        fold_validation = frame.loc[
            train_mask & frame["cv_fold"].eq(fold_index)
        ].copy()
        fold_train_dataset = TokenizedCommentDataset(
            fold_train,
            tokenizer,
            args.max_length,
            use_context,
            return_token_type_ids=return_token_type_ids,
        )
        fold_validation_dataset = TokenizedCommentDataset(
            fold_validation,
            tokenizer,
            args.max_length,
            use_context,
            return_token_type_ids=return_token_type_ids,
        )

        fold_seed = args.seed + fold_index
        set_seed(fold_seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABELS),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
        class_weights = calculate_class_weights(fold_train, args.class_weighting)
        updates_per_epoch = math.ceil(
            len(fold_train_dataset)
            / (args.train_batch_size * args.gradient_accumulation_steps)
        )
        estimated_total_steps = math.ceil(updates_per_epoch * args.epochs)
        warmup_steps = round(estimated_total_steps * args.warmup_ratio)

        fold_args = argparse.Namespace(**vars(args))
        fold_args.output_dir = fold_dir
        fold_args.seed = fold_seed
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": make_training_arguments(fold_args, warmup_steps),
            "train_dataset": fold_train_dataset,
            "eval_dataset": fold_validation_dataset,
            "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
            "compute_metrics": compute_metrics,
            "class_weights": class_weights,
        }
        trainer_signature = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = WeightedLossTrainer(**trainer_kwargs)
        train_result = trainer.train()
        model_dir = fold_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        trainer.save_state()
        with (fold_dir / "trainer_log_history.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(
                trainer.state.log_history,
                output,
                ensure_ascii=False,
                indent=2,
                default=json_ready,
            )

        validation_probs = predict_probabilities(trainer, fold_validation_dataset)
        validation_labels = fold_validation_dataset.frame["label_id"].to_numpy(
            dtype=np.int64
        )
        validation_predictions = validation_probs.argmax(axis=1)
        validation_metrics = metrics_with_confusion(
            validation_labels, validation_predictions
        )
        validation_output = prediction_frame(
            fold_validation_dataset,
            validation_probs,
            validation_predictions,
            validation_predictions,
        )
        validation_output.to_csv(
            fold_dir / "validation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # The fixed dev and test labels are not used here. Each fold only emits
        # probabilities; calibration and final scoring happen after all folds.
        calibration_probabilities.append(
            predict_probabilities(trainer, calibration_dataset)
        )
        test_probabilities.append(predict_probabilities(trainer, test_dataset))
        fold_results.append(
            {
                "fold": fold_number,
                "seed": fold_seed,
                "train_rows": len(fold_train_dataset),
                "validation_rows": len(fold_validation_dataset),
                "class_weights": class_weights.detach().cpu().tolist(),
                "estimated_total_steps": estimated_total_steps,
                "warmup_steps": warmup_steps,
                "train_metrics": train_result.metrics,
                "validation_metrics": validation_metrics,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "model_path": str(model_dir),
            }
        )

        del trainer, model, fold_train_dataset, fold_validation_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_mean, cv_std = aggregate_fold_metrics(fold_results)
    save_fold_metrics_csv(fold_results, args.output_dir / "fold_metrics.csv")

    calibration_probs = np.mean(np.stack(calibration_probabilities), axis=0)
    test_probs = np.mean(np.stack(test_probabilities), axis=0)
    calibration_labels = calibration_dataset.frame["label_id"].to_numpy(dtype=np.int64)
    test_labels = test_dataset.frame["label_id"].to_numpy(dtype=np.int64)
    raw_calibration_predictions = calibration_probs.argmax(axis=1)
    raw_test_predictions = test_probs.argmax(axis=1)
    dev_metrics_raw = metrics_with_confusion(
        calibration_labels, raw_calibration_predictions
    )

    thresholds, threshold_summary = tune_thresholds(
        calibration_labels,
        calibration_probs,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    calibrated_dev_predictions = threshold_predictions(calibration_probs, thresholds)
    calibrated_test_predictions = threshold_predictions(test_probs, thresholds)
    dev_metrics = metrics_with_confusion(
        calibration_labels, calibrated_dev_predictions
    )
    test_metrics = metrics_with_confusion(test_labels, calibrated_test_predictions)

    prediction_frame(
        calibration_dataset,
        calibration_probs,
        raw_calibration_predictions,
        calibrated_dev_predictions,
    ).to_csv(
        args.output_dir / "dev_threshold_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_frame(
        test_dataset,
        test_probs,
        raw_test_predictions,
        calibrated_test_predictions,
    ).to_csv(
        args.output_dir / "test_ensemble_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (args.output_dir / "selected_thresholds.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(threshold_summary, output, ensure_ascii=False, indent=2)

    best_fold = max(fold_results, key=lambda item: item["validation_metrics"]["macro_f1"])
    train_runtimes = [
        float(result["train_metrics"].get("train_runtime", 0.0))
        for result in fold_results
    ]
    train_losses = [
        float(result["train_metrics"]["train_loss"])
        for result in fold_results
        if result["train_metrics"].get("train_loss") is not None
    ]
    run_summary = {
        "status": "completed",
        "run_id": args.run_id,
        "experiment_name": args.experiment_name,
        "experiment_note": args.experiment_note,
        "started_at_kst": started_at_kst,
        "completed_at_kst": now_kst().isoformat(timespec="seconds"),
        "warning": (
            "Evaluation labels are LLM-generated silver labels. Test metrics are "
            "computed once after thresholds are fixed on the held-out dev split."
        ),
        "model_name": args.model_name,
        "label_source": "LLM silver",
        "source_to_stance": SOURCE_TO_STANCE,
        "label2id": LABEL2ID,
        "arguments": vars(args),
        "data_audit": data_audit,
        "split_summary": split_summary,
        "cv_summary": {
            "num_folds": args.num_folds,
            "scope": "train split only",
            "metrics_mean": cv_mean,
            "metrics_std": cv_std,
            "folds": fold_results,
        },
        "cv_metrics_mean": cv_mean,
        "cv_metrics_std": cv_std,
        "dev_metrics_raw": dev_metrics_raw,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "threshold_calibration": threshold_summary,
        "test_evaluation_count": 1,
        "train_metrics": {
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "train_runtime": float(np.sum(train_runtimes)),
            "total_wall_runtime": float(time.perf_counter() - total_started),
        },
        "best_model": {
            "fold": best_fold["fold"],
            "checkpoint": best_fold["best_checkpoint"],
            "metric": best_fold["validation_metrics"]["macro_f1"],
        },
        "environment": runtime_environment(),
    }
    return run_summary


def main() -> None:
    args = parse_args()
    args.run_id = args.run_id or make_run_id(args.seed)
    started_at_kst = now_kst().isoformat(timespec="seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_summary = run_experiment(args, started_at_kst)
    except Exception as error:
        print(
            f"Experiment {args.run_id} failed; run_summary.json was not written: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as output:
        json.dump(run_summary, output, ensure_ascii=False, indent=2, default=json_ready)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
