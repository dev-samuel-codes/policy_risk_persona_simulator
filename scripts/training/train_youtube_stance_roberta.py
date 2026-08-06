#!/usr/bin/env python3
"""Fine-tune a Korean encoder model for law/policy comment stance classification.

The input CSV is expected to contain LLM-generated silver labels. Only rows
whose ``relevance_label`` is ``관련 있음`` are used. The source
``sentiment_label`` is interpreted as policy stance using this fixed mapping:

    긍정 -> 찬성
    부정 -> 반대
    중립 -> 중립

Splits are made by connected groups of ``video_id`` values. Videos connected
by the same ``comment_hash`` are kept together as well, preventing video-level
and exact-comment leakage across train/dev/test.

This script is intended to run on the AMPM server, not on the local Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


SOURCE_TO_STANCE = {"긍정": "찬성", "부정": "반대", "중립": "중립"}
LABELS = ["찬성", "반대", "중립"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
REQUIRED_COLUMNS = {
    "comment_id",
    "comment_text",
    "video_id",
    "relevance_label",
    "sentiment_label",
}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    dev: float
    test: float

    def validate(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("All split ratios must be greater than zero")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("train/dev/test ratios must sum to 1.0")


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class TokenizedCommentDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        use_context: bool,
        return_token_type_ids: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.encodings = tokenizer(
            self.frame["comment_text"].tolist(),
            self.frame["context_text"].tolist() if use_context else None,
            truncation=True,
            max_length=max_length,
            padding=False,
            # KLUE RoBERTa has type_vocab_size=1, so pair token type IDs must be
            # disabled. ELECTRA supports them and benefits from distinguishing
            # the comment from its law/video context.
            return_token_type_ids=return_token_type_ids,
        )
        self.labels = self.frame["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Korean encoder on LLM-labeled YouTube stance data"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--experiment-name", default="youtube-stance-klue-roberta-base"
    )
    parser.add_argument(
        "--experiment-note",
        default="",
        help="Short hypothesis or note to include in run_summary.json",
    )
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Seed used only for the grouped train/dev/test split. Defaults to "
            "--seed for backward compatibility; set it explicitly when comparing "
            "multiple training seeds on an identical split"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-search-trials", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--text-mode",
        choices=("comment_context", "comment"),
        default="comment_context",
        help="Use comment plus law/video context, or the comment alone",
    )
    parser.add_argument(
        "--exclude-video-ids",
        type=Path,
        default=None,
        help="CSV containing video_id or a text file with one held-out video_id per line",
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run every configured epoch while still restoring the best dev checkpoint",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "sqrt_balanced"),
        default="sqrt_balanced",
    )
    parser.add_argument(
        "--train-support-to-oppose-ratio",
        type=float,
        default=None,
        help=(
            "Oversample support rows inside train only until support:oppose "
            "matches this ratio; dev/test remain untouched"
        ),
    )
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="Include rows where needs_review is true (excluded by default)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_excluded_video_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Excluded-video file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        if "video_id" not in frame.columns:
            raise ValueError("Excluded-video CSV must contain a video_id column")
        values = frame["video_id"].astype(str)
    else:
        values = pd.Series(path.read_text(encoding="utf-8-sig").splitlines(), dtype=str)
    return {value.strip() for value in values if value.strip()}


def build_context_text(frame: pd.DataFrame) -> pd.Series:
    context_fields = (
        ("topic_name", "주제"),
        ("law_name", "법률"),
        ("law_short_name", "법률 약칭"),
        ("video_title", "영상 제목"),
    )

    def build_row(row: pd.Series) -> str:
        parts = []
        for column, label in context_fields:
            value = str(row.get(column, "")).strip()
            if value:
                parts.append(f"{label}: {value}")
        return " | ".join(parts)

    return frame.apply(build_row, axis=1)


def load_silver_data(
    path: Path,
    include_needs_review: bool,
    excluded_video_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    source = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if source["comment_id"].duplicated().any():
        duplicates = int(source["comment_id"].duplicated(keep=False).sum())
        raise ValueError(f"comment_id must be unique; duplicated rows={duplicates}")

    frame = source.copy()
    frame["comment_text"] = frame["comment_text"].astype(str).str.strip()
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame["sentiment_label"] = frame["sentiment_label"].astype(str).str.strip()
    frame["relevance_label"] = frame["relevance_label"].astype(str).str.strip()

    valid = (
        frame["relevance_label"].eq("관련 있음")
        & frame["sentiment_label"].isin(SOURCE_TO_STANCE)
        & frame["comment_text"].ne("")
        & frame["video_id"].ne("")
    )
    if "needs_review" in frame.columns and not include_needs_review:
        valid &= ~frame["needs_review"].map(normalize_bool)
    if excluded_video_ids:
        valid &= ~frame["video_id"].isin(excluded_video_ids)

    selected = frame.loc[valid].copy()
    selected["stance_label"] = selected["sentiment_label"].map(SOURCE_TO_STANCE)
    selected["label_id"] = selected["stance_label"].map(LABEL2ID).astype(int)
    selected["context_text"] = build_context_text(selected)

    if "comment_hash" not in selected.columns:
        selected["comment_hash"] = selected["comment_text"].map(
            lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
    else:
        missing_hash = selected["comment_hash"].astype(str).str.strip().eq("")
        selected.loc[missing_hash, "comment_hash"] = selected.loc[
            missing_hash, "comment_text"
        ].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())

    if selected.empty:
        raise ValueError("No trainable rows remain after filtering")
    missing_labels = sorted(set(LABELS) - set(selected["stance_label"]))
    if missing_labels:
        raise ValueError(f"Trainable data is missing stance labels: {missing_labels}")

    audit = {
        "input_rows": int(len(source)),
        "selected_rows": int(len(selected)),
        "excluded_rows": int(len(source) - len(selected)),
        "include_needs_review": include_needs_review,
        "excluded_video_ids": sorted(excluded_video_ids),
        "excluded_video_count": len(excluded_video_ids),
        "selected_label_counts": {
            key: int(value)
            for key, value in selected["stance_label"].value_counts().items()
        },
        "selected_videos": int(selected["video_id"].nunique()),
        "input_sha256": sha256_file(path),
    }
    return selected, audit


def add_leakage_groups(frame: pd.DataFrame) -> pd.DataFrame:
    videos = sorted(frame["video_id"].unique())
    union_find = UnionFind(videos)

    shared_hashes = frame.groupby("comment_hash")["video_id"].unique()
    for video_ids in shared_hashes:
        if len(video_ids) < 2:
            continue
        first = str(video_ids[0])
        for video_id in video_ids[1:]:
            union_find.union(first, str(video_id))

    grouped = frame.copy()
    grouped["leakage_group"] = grouped["video_id"].map(union_find.find)
    return grouped


def make_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    row_counts = frame.groupby("leakage_group").size().rename("rows")
    class_counts = pd.crosstab(frame["leakage_group"], frame["label_id"]).reindex(
        columns=range(len(LABELS)), fill_value=0
    )
    class_counts.columns = [f"label_{label_id}" for label_id in class_counts.columns]
    return row_counts.to_frame().join(class_counts).fillna(0).astype(int)


def split_score(
    group_summary: pd.DataFrame,
    assignment: dict[str, str],
    ratios: SplitRatios,
) -> float | None:
    split_names = ("train", "dev", "test")
    ratio_by_split = asdict(ratios)
    total_rows = int(group_summary["rows"].sum())
    label_columns = [f"label_{label_id}" for label_id in range(len(LABELS))]
    total_class_share = group_summary[label_columns].sum().to_numpy() / total_rows
    assigned_split = pd.Series(assignment)
    score = 0.0
    for split in split_names:
        selected_groups = assigned_split.index[assigned_split.eq(split)]
        if selected_groups.empty:
            return None
        part = group_summary.loc[selected_groups]
        class_counts = part[label_columns].sum().to_numpy()
        if np.any(class_counts == 0):
            return None
        part_rows = int(part["rows"].sum())
        row_share = part_rows / total_rows
        score += 5.0 * (row_share - ratio_by_split[split]) ** 2
        class_share = class_counts / part_rows
        score += float(np.square(class_share - total_class_share).sum())
    return score


def make_grouped_splits(
    frame: pd.DataFrame,
    ratios: SplitRatios,
    seed: int,
    trials: int,
) -> pd.DataFrame:
    ratios.validate()
    groups = sorted(frame["leakage_group"].unique())
    if len(groups) < 3:
        raise ValueError("At least three independent leakage groups are required")

    rng = random.Random(seed)
    group_summary = make_group_summary(frame)
    split_names = ("train", "dev", "test")
    weights = (ratios.train, ratios.dev, ratios.test)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    for _ in range(trials):
        assignment = {
            group: rng.choices(split_names, weights=weights, k=1)[0] for group in groups
        }
        score = split_score(group_summary, assignment, ratios)
        if score is not None and score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError(
            "Could not create a group-safe split containing all labels. "
            "Increase --split-search-trials or inspect label distribution by video."
        )

    result = frame.copy()
    result["split"] = result["leakage_group"].map(best_assignment)
    validate_no_leakage(result)
    return result


def validate_no_leakage(frame: pd.DataFrame) -> None:
    for column in ("video_id", "comment_hash", "comment_id", "leakage_group"):
        counts = frame.groupby(column)["split"].nunique()
        leaked = counts[counts > 1]
        if not leaked.empty:
            raise AssertionError(
                f"Leakage detected for {column}: {len(leaked)} values cross splits"
            )


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {"accuracy": float(np.mean(labels == predictions))}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    metric_names = {"찬성": "support", "반대": "oppose", "중립": "neutral"}
    for label_id, label_name in ID2LABEL.items():
        true_positive = int(np.sum((labels == label_id) & (predictions == label_id)))
        false_positive = int(np.sum((labels != label_id) & (predictions == label_id)))
        false_negative = int(np.sum((labels == label_id) & (predictions != label_id)))
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
        key = metric_names[label_name]
        result[f"precision_{key}"] = precision
        result[f"recall_{key}"] = recall
        result[f"f1_{key}"] = f1
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    result["macro_precision"] = float(np.mean(precisions))
    result["macro_recall"] = float(np.mean(recalls))
    result["macro_f1"] = float(np.mean(f1_scores))
    return result


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = np.asarray(eval_prediction.label_ids)
    predictions = np.argmax(np.asarray(logits), axis=-1)
    return classification_metrics(labels, predictions)


def calculate_class_weights(frame: pd.DataFrame, mode: str) -> torch.Tensor:
    counts = (
        frame["label_id"]
        .value_counts()
        .reindex(range(len(LABELS)), fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    if np.any(counts == 0):
        raise ValueError(f"Training split contains an empty class: {counts.tolist()}")
    if mode == "none":
        weights = np.ones_like(counts)
    else:
        balanced = len(frame) / (len(LABELS) * counts)
        weights = np.sqrt(balanced) if mode == "sqrt_balanced" else balanced
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def oversample_training_frame(
    frame: pd.DataFrame,
    support_to_oppose_ratio: float | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_counts = {
        key: int(value) for key, value in frame["stance_label"].value_counts().items()
    }
    if support_to_oppose_ratio is None:
        return frame.copy(), {
            "enabled": False,
            "before_rows": int(len(frame)),
            "before_label_counts": before_counts,
        }
    if support_to_oppose_ratio <= 0:
        raise ValueError("--train-support-to-oppose-ratio must be greater than zero")
    if "찬성" not in LABEL2ID or "반대" not in LABEL2ID:
        raise ValueError("Support:oppose oversampling requires 찬성 and 반대 labels")

    support = frame.loc[frame["stance_label"].eq("찬성")]
    oppose = frame.loc[frame["stance_label"].eq("반대")]
    if support.empty or oppose.empty:
        raise ValueError("Training split must contain both 찬성 and 반대 rows")
    other_rows = frame.loc[~frame["stance_label"].isin(("찬성", "반대"))]
    if not other_rows.empty:
        raise ValueError(
            "Support:oppose oversampling is only supported for the binary task"
        )

    desired_support_rows = int(math.ceil(len(oppose) * support_to_oppose_ratio))
    additional_rows = desired_support_rows - len(support)
    if additional_rows < 0:
        current_ratio = len(support) / len(oppose)
        raise ValueError(
            "Requested support:oppose ratio is below the current train ratio; "
            f"oversampling cannot reduce rows (current={current_ratio:.6f})"
        )

    original = frame.copy()
    original["oversampled_copy"] = False
    original["oversample_source_index"] = original.index.astype(int)
    if additional_rows:
        copies = support.sample(
            n=additional_rows,
            replace=True,
            random_state=seed,
        ).copy()
        copies["oversampled_copy"] = True
        copies["oversample_source_index"] = copies.index.astype(int)
        sampled = pd.concat([original, copies], ignore_index=True)
    else:
        sampled = original.reset_index(drop=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled["training_sample_index"] = np.arange(len(sampled), dtype=np.int64)

    after_counts = {
        key: int(value) for key, value in sampled["stance_label"].value_counts().items()
    }
    actual_ratio = after_counts["찬성"] / after_counts["반대"]
    audit = {
        "enabled": True,
        "scope": "train_only",
        "method": "random_oversampling_with_replacement",
        "seed": seed,
        "requested_support_to_oppose_ratio": support_to_oppose_ratio,
        "actual_support_to_oppose_ratio": actual_ratio,
        "before_rows": int(len(frame)),
        "after_rows": int(len(sampled)),
        "added_support_copies": int(additional_rows),
        "before_label_counts": before_counts,
        "after_label_counts": after_counts,
        "dev_test_oversampled": False,
    }
    return sampled, audit


def make_training_arguments(
    args: argparse.Namespace,
    warmup_steps: int,
) -> TrainingArguments:
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "linear",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": use_cuda,
        "fp16": bool(use_cuda and not use_bf16),
        "bf16": use_bf16,
        "tf32": bool(use_cuda),
        "report_to": "none",
        "remove_unused_columns": True,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs[strategy_key] = "epoch"
    return TrainingArguments(**kwargs)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions, strict=True):
        matrix[int(actual), int(predicted)] += 1
    return matrix.tolist()


def save_predictions(
    trainer: Trainer,
    dataset: TokenizedCommentDataset,
    output_path: Path,
) -> dict[str, Any]:
    prediction = trainer.predict(dataset)
    logits = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted_ids = probabilities.argmax(axis=-1)
    actual_ids = dataset.frame["label_id"].to_numpy(dtype=np.int64)

    output_columns = [
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
            output_columns.append(optional_column)
    output = dataset.frame[output_columns].copy()
    output["predicted_label"] = [ID2LABEL[int(value)] for value in predicted_ids]
    output["correct"] = actual_ids == predicted_ids
    for label_id, label_name in ID2LABEL.items():
        output[f"prob_{label_name}"] = probabilities[:, label_id]
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    metrics = classification_metrics(actual_ids, predicted_ids)
    metrics["confusion_matrix_actual_rows_predicted_columns"] = confusion_matrix(
        actual_ids, predicted_ids
    )
    return metrics


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def make_run_id(seed: int) -> str:
    timestamp = now_kst().strftime("%Y%m%d_%H%M%S_KST")
    return f"{timestamp}_seed{seed}_pid{os.getpid()}"


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return markdown_cell(value)


def format_mean_std(mean: Any, std: Any, digits: int = 4) -> str:
    if mean is None:
        return "-"
    if std is None:
        return format_number(mean, digits)
    return f"{format_number(mean, digits)} ± {format_number(std, digits)}"


def describe_training_method(summary: dict[str, Any]) -> str:
    args = summary.get("arguments", {})
    model_name = str(summary.get("model_name", args.get("model_name", "")))
    architecture = "KcELECTRA" if "kcelectra" in model_name.lower() else "RoBERTa"
    text_mode = {
        "comment_context": "댓글+법률·영상 문맥",
        "comment": "댓글 단독",
    }.get(args.get("text_mode"), args.get("text_mode", "-"))
    cv_prefix = ""
    if int(args.get("num_folds", 1) or 1) > 1:
        cv_prefix = (
            f"{args.get('num_folds')}-fold CV · fold당 "
            f"{format_number(args.get('epochs'), 0)} epoch · "
        )
    return (
        f"{architecture} 전체 파인튜닝 · LLM silver 3분류 · {text_mode} · "
        f"{cv_prefix}{args.get('class_weighting', '-')} 가중 손실"
    )


def render_confusion_matrix(matrix: Any) -> list[str]:
    if not matrix:
        return ["- 기록 없음"]
    lines = [
        "| 실제 \\ 예측 | 찬성 | 반대 | 중립 |",
        "|---|---:|---:|---:|",
    ]
    for label, row in zip(LABELS, matrix, strict=True):
        values = " | ".join(str(int(value)) for value in row)
        lines.append(f"| {label} | {values} |")
    return lines


def render_metric_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    if summary.get("cv_metrics_mean"):
        mean = summary["cv_metrics_mean"]
        std = summary.get("cv_metrics_std", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    "5-fold 내부 검증 평균±표준편차",
                    format_mean_std(mean.get("accuracy"), std.get("accuracy")),
                    format_mean_std(
                        mean.get("macro_precision"), std.get("macro_precision")
                    ),
                    format_mean_std(mean.get("macro_recall"), std.get("macro_recall")),
                    format_mean_std(mean.get("macro_f1"), std.get("macro_f1")),
                ]
            )
            + " |"
        )
    metric_splits = (
        ("dev_metrics", "임계값 설정용 dev")
        if summary.get("cv_metrics_mean")
        else ("dev_metrics", "dev")
    ), (
        "test_metrics",
        "최종 test" if summary.get("cv_metrics_mean") else "test",
    )
    for split_name, display_name in metric_splits:
        metrics = summary.get(split_name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    format_number(metrics.get("accuracy")),
                    format_number(metrics.get("macro_precision")),
                    format_number(metrics.get("macro_recall")),
                    format_number(metrics.get("macro_f1")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| 평가 데이터 | 클래스 | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metric_names = {"찬성": "support", "반대": "oppose", "중립": "neutral"}
    if summary.get("cv_metrics_mean"):
        mean = summary["cv_metrics_mean"]
        std = summary.get("cv_metrics_std", {})
        metric_names = {"찬성": "support", "반대": "oppose", "중립": "neutral"}
        for label in LABELS:
            key = metric_names[label]
            lines.append(
                "| "
                + " | ".join(
                    [
                        "5-fold 내부 검증 평균±표준편차",
                        label,
                        format_mean_std(
                            mean.get(f"precision_{key}"), std.get(f"precision_{key}")
                        ),
                        format_mean_std(
                            mean.get(f"recall_{key}"), std.get(f"recall_{key}")
                        ),
                        format_mean_std(mean.get(f"f1_{key}"), std.get(f"f1_{key}")),
                    ]
                )
                + " |"
            )
    for split_name, display_name in metric_splits:
        metrics = summary.get(split_name, {})
        for label in LABELS:
            key = metric_names[label]
            lines.append(
                "| "
                + " | ".join(
                    [
                        display_name,
                        label,
                        format_number(metrics.get(f"precision_{key}")),
                        format_number(metrics.get(f"recall_{key}")),
                        format_number(metrics.get(f"f1_{key}")),
                    ]
                )
                + " |"
            )
    return lines


def render_experiment_markdown(summary: dict[str, Any]) -> str:
    status = summary.get("status", "unknown")
    status_korean = {"completed": "완료", "failed": "실패"}.get(status, status)
    args = summary.get("arguments", {})
    lines = [
        f"## {summary.get('run_id', 'unknown')} — {status_korean}",
        "",
    ]

    if status == "completed":
        training_method = describe_training_method(summary)
        lines.extend(
            [
                "### 실험 요약",
                "",
                "| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        if summary.get("cv_metrics_mean"):
            mean = summary["cv_metrics_mean"]
            std = summary.get("cv_metrics_std", {})
            lines.append(
                f"| {training_method} | 5-fold 내부 검증 평균±표준편차 | "
                f"{format_mean_std(mean.get('accuracy'), std.get('accuracy'))} | "
                f"{format_mean_std(mean.get('macro_precision'), std.get('macro_precision'))} | "
                f"{format_mean_std(mean.get('macro_recall'), std.get('macro_recall'))} | "
                f"{format_mean_std(mean.get('macro_f1'), std.get('macro_f1'))} |"
            )
        summary_splits = (
            ("dev_metrics", "임계값 설정용 검증(dev)"),
            ("test_metrics", "최종 테스트(test)"),
        ) if summary.get("cv_metrics_mean") else (
            ("dev_metrics", "검증(dev)"),
            ("test_metrics", "테스트(test)"),
        )
        for split_name, display_name in summary_splits:
            metrics = summary.get(split_name, {})
            lines.append(
                f"| {training_method} | {display_name} | "
                f"{format_number(metrics.get('accuracy'))} | "
                f"{format_number(metrics.get('macro_precision'))} | "
                f"{format_number(metrics.get('macro_recall'))} | "
                f"{format_number(metrics.get('macro_f1'))} |"
            )
        lines.append("")

    lines.extend(
        [
            f"- 실험명: `{markdown_cell(summary.get('experiment_name', '-'))}`",
            f"- 시작: {markdown_cell(summary.get('started_at_kst', '-'))}",
            f"- 종료: {markdown_cell(summary.get('completed_at_kst', '-'))}",
            f"- 모델: `{markdown_cell(summary.get('model_name', args.get('model_name', '-')))}`",
            f"- 출력 폴더: `{markdown_cell(args.get('output_dir', '-'))}`",
            f"- 메모: {markdown_cell(summary.get('experiment_note') or '-')}",
            "",
        ]
    )

    if status == "failed":
        lines.extend(
            [
                "### 실패 정보",
                "",
                f"- 예외: `{markdown_cell(summary.get('error_type', '-'))}`",
                f"- 메시지: {markdown_cell(summary.get('error_message', '-'))}",
                "",
                "```text",
                str(summary.get("traceback", "기록 없음")).rstrip(),
                "```",
                "",
                "### 실행 설정",
                "",
                "```json",
                json.dumps(args, ensure_ascii=False, indent=2, default=json_ready),
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    data_audit = summary.get("data_audit", {})
    split_summary = summary.get("split_summary", {})
    train_metrics = summary.get("train_metrics", {})
    environment = summary.get("environment", {})
    best = summary.get("best_model", {})
    if summary.get("cv_metrics_mean"):
        one_line = (
            f"5-fold 내부 검증 Macro-F1 "
            f"`{format_mean_std(summary.get('cv_metrics_mean', {}).get('macro_f1'), summary.get('cv_metrics_std', {}).get('macro_f1'))}`, "
            f"임계값 설정용 dev Macro-F1 `{format_number(summary.get('dev_metrics', {}).get('macro_f1'))}`, "
            f"최종 test Macro-F1 `{format_number(summary.get('test_metrics', {}).get('macro_f1'))}`; "
            "LLM silver 라벨에 대한 일치 성능입니다."
        )
    else:
        one_line = (
            f"dev Macro-F1 `{format_number(summary.get('dev_metrics', {}).get('macro_f1'))}`, "
            f"test Macro-F1 `{format_number(summary.get('test_metrics', {}).get('macro_f1'))}`; "
            "LLM silver 라벨에 대한 일치 성능입니다."
        )
    lines.extend(
        [
            "### 한줄 요약",
            "",
            one_line,
            "",
            "### 데이터",
            "",
            f"- 입력 SHA-256: `{markdown_cell(data_audit.get('input_sha256', '-'))}`",
            f"- 원본/선택/제외 행: {data_audit.get('input_rows', '-')} / "
            f"{data_audit.get('selected_rows', '-')} / {data_audit.get('excluded_rows', '-')}",
            f"- 선택 영상 수: {data_audit.get('selected_videos', '-')}",
            f"- `needs_review=true` 포함: {data_audit.get('include_needs_review', '-')}",
            "",
            "| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 중립 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "dev", "test"):
        item = split_summary.get(split, {})
        labels = item.get("labels", {})
        lines.append(
            f"| {split} | {item.get('rows', '-')} | {item.get('videos', '-')} | "
            f"{item.get('leakage_groups', '-')} | {labels.get('찬성', 0)} | "
            f"{labels.get('반대', 0)} | {labels.get('중립', 0)} |"
        )

    lines.extend(
        [
            "",
            "### 주요 설정",
            "",
            "| 항목 | 값 |",
            "|---|---|",
        ]
    )
    settings = (
        ("seed", "seed"),
        ("folds", "num_folds"),
        ("text mode", "text_mode"),
        ("token type IDs", "return_token_type_ids"),
        ("max length", "max_length"),
        ("epochs", "epochs"),
        ("train batch", "train_batch_size"),
        ("eval batch", "eval_batch_size"),
        ("gradient accumulation", "gradient_accumulation_steps"),
        ("learning rate", "learning_rate"),
        ("weight decay", "weight_decay"),
        ("warmup ratio", "warmup_ratio"),
        ("class weighting", "class_weighting"),
    )
    for display_name, key in settings:
        lines.append(f"| {display_name} | {markdown_cell(args.get(key, '-'))} |")

    if summary.get("cv_metrics_mean"):
        threshold = summary.get("threshold_calibration", {})
        selected_thresholds = threshold.get("selected", {})
        lines.extend(
            [
                "",
                "### 임계값 설정",
                "",
                "- 검증(dev) 10%에서만 Macro-F1 기준으로 설정",
                f"- 결정 규칙: `{markdown_cell(threshold.get('decision_rule', '-'))}`",
                f"- 찬성/반대/중립 임계값: "
                f"`{format_number(selected_thresholds.get('찬성'))}` / "
                f"`{format_number(selected_thresholds.get('반대'))}` / "
                f"`{format_number(selected_thresholds.get('중립'))}`",
                f"- 테스트 평가 횟수: {summary.get('test_evaluation_count', '-')}",
            ]
        )

    lines.extend(
        [
            "",
            "### 결과",
            "",
            *render_metric_table(summary),
            "",
            f"- train loss: {format_number(train_metrics.get('train_loss'))}",
            f"- train runtime(초): {format_number(train_metrics.get('train_runtime'), 2)}",
            f"- best checkpoint: `{markdown_cell(best.get('checkpoint', '-'))}`",
            f"- best dev metric: {format_number(best.get('metric'))}",
            "",
            "#### Dev 혼동행렬",
            "",
            *render_confusion_matrix(
                summary.get("dev_metrics", {}).get(
                    "confusion_matrix_actual_rows_predicted_columns"
                )
            ),
            "",
            "#### Test 혼동행렬",
            "",
            *render_confusion_matrix(
                summary.get("test_metrics", {}).get(
                    "confusion_matrix_actual_rows_predicted_columns"
                )
            ),
            "",
            "### 환경 및 산출물",
            "",
            f"- Python: `{markdown_cell(environment.get('python', '-'))}`",
            f"- PyTorch: `{markdown_cell(environment.get('torch', '-'))}`",
            f"- Transformers: `{markdown_cell(environment.get('transformers', '-'))}`",
            f"- CUDA: `{markdown_cell(environment.get('cuda_version', '-'))}`",
            f"- GPU: `{markdown_cell(', '.join(environment.get('gpu_names', [])) or '-')}`",
            f"- 모델: `{markdown_cell(Path(args.get('output_dir', '-')) / ('fold_*/model' if summary.get('cv_metrics_mean') else 'model'))}`",
            f"- JSON 요약: `{markdown_cell(Path(args.get('output_dir', '-')) / 'run_summary.json')}`",
            f"- 분할표: `{markdown_cell(Path(args.get('output_dir', '-')) / 'split_manifest.csv')}`",
            f"- dev 예측: `{markdown_cell(Path(args.get('output_dir', '-')) / ('dev_threshold_predictions.csv' if summary.get('cv_metrics_mean') else 'dev_predictions.csv'))}`",
            f"- test 예측: `{markdown_cell(Path(args.get('output_dir', '-')) / ('test_ensemble_predictions.csv' if summary.get('cv_metrics_mean') else 'test_predictions.csv'))}`",
            "",
            "> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def run_experiment(
    args: argparse.Namespace,
    started_at_kst: str,
) -> dict[str, Any]:
    ratios = SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio)
    ratios.validate()
    if args.split_search_trials < 1:
        raise ValueError("--split-search-trials must be at least 1")
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    excluded_video_ids = load_excluded_video_ids(args.exclude_video_ids)
    frame, data_audit = load_silver_data(
        args.data,
        args.include_needs_review,
        excluded_video_ids,
    )
    frame = add_leakage_groups(frame)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    frame = make_grouped_splits(
        frame,
        ratios=ratios,
        seed=split_seed,
        trials=args.split_search_trials,
    )
    data_audit["split_seed"] = split_seed

    manifest_columns = [
        "comment_id",
        "video_id",
        "comment_hash",
        "leakage_group",
        "sentiment_label",
        "stance_label",
        "label_id",
        "split",
    ]
    frame[manifest_columns].to_csv(
        args.output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    if "law_original_text" in frame.columns:
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
        ]
        frame[feature_manifest_columns].to_csv(
            args.output_dir / "feature_context_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    split_summary = {}
    for split_name in ("train", "dev", "test"):
        part = frame.loc[frame["split"].eq(split_name)]
        split_summary[split_name] = {
            "rows": int(len(part)),
            "videos": int(part["video_id"].nunique()),
            "leakage_groups": int(part["leakage_group"].nunique()),
            "labels": {
                key: int(value)
                for key, value in part["stance_label"].value_counts().items()
            },
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    # Transformers 5.x TokenizersBackend can emit token_type_ids for
    # KcELECTRA sentence pairs even when model_input_names omits the field.
    # The model config is therefore the reliable architecture capability flag.
    return_token_type_ids = bool(getattr(model.config, "type_vocab_size", 1) > 1)
    args.return_token_type_ids = return_token_type_ids
    train_frame, training_sampling = oversample_training_frame(
        frame.loc[frame["split"].eq("train")].copy(),
        args.train_support_to_oppose_ratio,
        args.seed,
    )
    if training_sampling["enabled"]:
        sampling_manifest_columns = [
            "training_sample_index",
            "oversample_source_index",
            "oversampled_copy",
            "comment_id",
            "video_id",
            "stance_label",
            "label_id",
        ]
        train_frame[sampling_manifest_columns].to_csv(
            args.output_dir / "training_sampling_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
    dataset_frames = {
        "train": train_frame,
        "dev": frame.loc[frame["split"].eq("dev")].copy(),
        "test": frame.loc[frame["split"].eq("test")].copy(),
    }
    datasets = {
        split: TokenizedCommentDataset(
            dataset_frame,
            tokenizer,
            args.max_length,
            use_context=args.text_mode == "comment_context",
            return_token_type_ids=return_token_type_ids,
        )
        for split, dataset_frame in dataset_frames.items()
    }
    class_weights = calculate_class_weights(datasets["train"].frame, args.class_weighting)
    updates_per_epoch = math.ceil(
        len(datasets["train"])
        / (args.train_batch_size * args.gradient_accumulation_steps)
    )
    estimated_total_steps = math.ceil(updates_per_epoch * args.epochs)
    warmup_steps = round(estimated_total_steps * args.warmup_ratio)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": make_training_arguments(args, warmup_steps),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["dev"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
    }
    if not args.disable_early_stopping:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        ]
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedLossTrainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(args.output_dir / "model")
    trainer.save_state()
    with (args.output_dir / "trainer_log_history.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(trainer.state.log_history, output, ensure_ascii=False, indent=2)

    dev_metrics = save_predictions(
        trainer, datasets["dev"], args.output_dir / "dev_predictions.csv"
    )
    test_metrics = save_predictions(
        trainer, datasets["test"], args.output_dir / "test_predictions.csv"
    )

    run_summary = {
        "status": "completed",
        "run_id": args.run_id,
        "experiment_name": args.experiment_name,
        "experiment_note": args.experiment_note,
        "started_at_kst": started_at_kst,
        "completed_at_kst": now_kst().isoformat(timespec="seconds"),
        "warning": (
            "Evaluation labels are LLM-generated silver labels. Metrics measure "
            "agreement with the labeling LLM, not verified real-world stance accuracy."
        ),
        "model_name": args.model_name,
        "label_source": "LLM silver",
        "source_to_stance": SOURCE_TO_STANCE,
        "label2id": LABEL2ID,
        "arguments": vars(args),
        "data_audit": data_audit,
        "split_summary": split_summary,
        "training_sampling": training_sampling,
        "class_weights": class_weights,
        "estimated_total_steps": estimated_total_steps,
        "warmup_steps": warmup_steps,
        "train_metrics": train_result.metrics,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "best_model": {
            "checkpoint": trainer.state.best_model_checkpoint,
            "metric": trainer.state.best_metric,
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
