#!/usr/bin/env python3
"""고정된 공개 FAQ 쌍을 현재 민원 매칭 기준으로 평가한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (  # noqa: E402
    FAQ_DATA_DIR,
    build_civil_complaint_search_document,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
    normalize_text,
)
from backend.ai_simulation_core.complaints.civil_complaint_similarity import (  # noqa: E402
    COMPLAINT_DENSE_FLOOR,
    DEFAULT_MODEL,
    SCORE_EPSILON,
    TOPIC_LEXICAL_FLOOR,
    _topic_overlap_evidence,
    lexical_score,
)


DEFAULT_FIXTURE = (
    PROJECT_ROOT
    / "backend"
    / "tests"
    / "fixtures"
    / "civil_complaint_matching_eval.json"
)
SOURCE_KEYS = (
    "detail_sha256",
    "metadata_sha256",
    "raw_record_count",
    "unique_count",
)
FALLBACK_BASES = {"lexical_fallback", "no_core_topic_overlap"}


class EvaluationError(ValueError):
    """평가셋이나 FAQ 정본을 신뢰할 수 없을 때 발생한다."""


def _load_fixture(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"평가 fixture를 읽을 수 없습니다: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise EvaluationError("평가 fixture schema_version은 1이어야 합니다.")
    if not isinstance(payload.get("source"), dict):
        raise EvaluationError("평가 fixture source가 없습니다.")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 32:
        raise EvaluationError("평가 fixture에는 최소 32개 pair가 필요합니다.")
    return payload


def _validate_pairs(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    fingerprint = civil_complaint_source_fingerprint(FAQ_DATA_DIR)
    for key in SOURCE_KEYS:
        if payload["source"].get(key) != fingerprint.get(key):
            raise EvaluationError(f"FAQ 정본 fingerprint가 다릅니다: {key}")

    corpus = load_civil_complaint_corpus(FAQ_DATA_DIR)
    records = {normalize_text(row.get("case_id")): row for row in corpus}
    if len(records) != len(corpus) or "" in records:
        raise EvaluationError("FAQ 정본 case_id가 비어 있거나 중복됩니다.")

    required_strings = (
        "pair_id",
        "query_group",
        "domain",
        "query_text",
        "case_id",
        "candidate_title_snapshot",
    )
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    case_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_pair in enumerate(payload["pairs"]):
        if not isinstance(raw_pair, dict):
            raise EvaluationError(f"pairs[{index}]는 객체여야 합니다.")
        pair = dict(raw_pair)
        for key in required_strings:
            value = pair.get(key)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"pairs[{index}].{key}가 비어 있습니다.")
            pair[key] = value.strip()
        if not isinstance(pair.get("expected_match"), bool):
            raise EvaluationError(f"pairs[{index}].expected_match가 boolean이 아닙니다.")
        if pair["pair_id"] in pair_ids or pair["case_id"] in case_ids:
            raise EvaluationError("pair_id 또는 case_id가 중복됩니다.")
        pair_ids.add(pair["pair_id"])
        case_ids.add(pair["case_id"])

        record = records.get(pair["case_id"])
        if record is None:
            raise EvaluationError(f"FAQ 정본에 없는 case_id입니다: {pair['case_id']}")
        if normalize_text(record.get("title")) != pair["candidate_title_snapshot"]:
            raise EvaluationError(f"FAQ 제목 snapshot이 다릅니다: {pair['pair_id']}")
        groups[pair["query_group"]].append(pair)
        pairs.append(pair)

    if sum(pair["expected_match"] for pair in pairs) * 2 != len(pairs):
        raise EvaluationError("positive와 negative pair 수가 같아야 합니다.")
    for name, rows in groups.items():
        if len(rows) != 2 or {row["expected_match"] for row in rows} != {True, False}:
            raise EvaluationError(f"query_group 구성이 올바르지 않습니다: {name}")
        if len({row["query_text"] for row in rows}) != 1:
            raise EvaluationError(f"query_group의 query_text가 다릅니다: {name}")
    return pairs, records, fingerprint


def _encode(
    texts: list[str], *, device: str, batch_size: int, local_files_only: bool
) -> tuple[np.ndarray, str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            DEFAULT_MODEL,
            device=device,
            local_files_only=local_files_only,
        )
        model.max_seq_length = 512
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as error:
        raise EvaluationError(f"임베딩 모델 실행에 실패했습니다: {error}") from error
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise EvaluationError(f"임베딩 행렬이 올바르지 않습니다: {matrix.shape}")
    return matrix, str(getattr(model, "device", device))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["expected_match"] and row["predicted_match"] for row in rows)
    fp = sum(not row["expected_match"] and row["predicted_match"] for row in rows)
    tn = sum(not row["expected_match"] and not row["predicted_match"] for row in rows)
    fn = sum(row["expected_match"] and not row["predicted_match"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "support": len(rows),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def _evaluate(
    pairs: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts: list[str] = []
    positions: dict[str, int] = {}

    def position(text: str) -> int:
        if text not in positions:
            positions[text] = len(texts)
            texts.append(text)
        return positions[text]

    pair_positions = []
    for pair in pairs:
        record = records[pair["case_id"]]
        pair_positions.append(
            (
                position(pair["query_text"]),
                position(build_civil_complaint_search_document(record)),
            )
        )
    embeddings, resolved_device = _encode(
        texts,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )

    results = []
    for pair, (query_position, candidate_position) in zip(
        pairs, pair_positions, strict=True
    ):
        record = records[pair["case_id"]]
        dense = float(np.dot(embeddings[query_position], embeddings[candidate_position]))
        dense = max(0.0, min(1.0, dense))
        heading = f"{record['title']} {record['question']}"
        lexical = lexical_score(pair["query_text"], heading)
        topic_matches, topic_evidence = _topic_overlap_evidence(
            pair["query_text"], heading, lexical=lexical
        )
        predicted = bool(
            dense + SCORE_EPSILON >= COMPLAINT_DENSE_FLOOR and topic_matches
        )
        basis = normalize_text(topic_evidence.get("basis"))
        results.append(
            {
                "pair_id": pair["pair_id"],
                "query_group": pair["query_group"],
                "domain": pair["domain"],
                "case_id": pair["case_id"],
                "expected_match": pair["expected_match"],
                "predicted_match": predicted,
                "correct": predicted == pair["expected_match"],
                "fallback_path": basis in FALLBACK_BASES,
                "complaint_dense": round(dense, 6),
                "lexical": round(float(lexical), 6),
                "topic_evidence": topic_evidence,
            }
        )
    return results, {
        "name": DEFAULT_MODEL,
        "device": resolved_device,
        "embedding_dimension": int(embeddings.shape[1]),
        "normalized_embeddings": True,
    }


def _report(
    payload: dict[str, Any],
    fingerprint: dict[str, Any],
    results: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    by_domain = {
        domain: _metrics([row for row in results if row["domain"] == domain])
        for domain in sorted({row["domain"] for row in results})
    }
    fallback_rows = [row for row in results if row["fallback_path"]]
    return {
        "status": "ok",
        "dataset_id": payload["dataset_id"],
        "source_fingerprint": fingerprint,
        "model": model,
        "thresholds": {
            "complaint_dense_floor": COMPLAINT_DENSE_FLOOR,
            "topic_lexical_floor": TOPIC_LEXICAL_FLOOR,
            "policy": "report_only_no_automatic_threshold_change",
        },
        "metrics": {
            "overall": _metrics(results),
            "by_domain": by_domain,
            "fallback": _metrics(fallback_rows),
        },
        "mismatches": [row["pair_id"] for row in results if not row["correct"]],
        "pairs": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="고정된 관련·무관 FAQ 쌍을 현재 dense/topic 기준으로 평가합니다."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--device",
        default=os.getenv("CIVIL_COMPLAINT_EMBEDDING_DEVICE", "cpu") or "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size < 1:
        print(
            '{"status":"error","message":"batch-size는 1 이상이어야 합니다."}',
            file=sys.stderr,
        )
        return 2
    try:
        payload = _load_fixture(args.fixture.expanduser())
        pairs, records, fingerprint = _validate_pairs(payload)
        results, model = _evaluate(
            pairs,
            records,
            device=str(args.device),
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        report = _report(payload, fingerprint, results, model)
    except EvaluationError as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
