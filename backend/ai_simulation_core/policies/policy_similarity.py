import json
import os
import re
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.ai_simulation_core.policies.policy_corpus import (
    PROJECT_ROOT,
    build_direct_policy_query,
    build_policy_search_document,
    date_ordinal,
    load_policy_corpus,
    normalize_date,
    normalize_text,
)

DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "policies" / "current"
DEFAULT_COLLECTION_NAME = "policy_similarity"
DEFAULT_MIN_SCORE = 0.32

TOKEN_PATTERN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[가-힣A-Za-z]{2,}")
CONDITION_PATTERN = re.compile(
    r"\d[\d,.]*\s*(?:세|원|만원|억원|퍼센트|%|개월|년|인|명|가구|회)"
)
STOP_WORDS = {
    "관련",
    "경우",
    "대상",
    "대한",
    "등을",
    "또는",
    "서비스",
    "신청",
    "있는",
    "위한",
    "지원",
    "정책",
    "통해",
    "하는",
    "하여",
    "해당",
}
FIELD_LABELS = {
    "policy_name": "정책명",
    "purpose": "정책목적",
    "category": "서비스분야",
    "support_type": "지원유형",
    "target_audience": "지원대상",
    "selection_criteria": "선정기준",
    "benefits": "지원내용",
    "application_period": "신청기한",
    "application_method": "신청방법",
}
FIELD_WEIGHTS = {
    "policy_name": 0.20,
    "purpose": 0.10,
    "category": 0.05,
    "support_type": 0.05,
    "target_audience": 0.25,
    "selection_criteria": 0.10,
    "benefits": 0.20,
    "application_period": 0.025,
    "application_method": 0.025,
}


class PolicyIndexUnavailableError(RuntimeError):
    pass


def tokenize(value: object) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(normalize_text(value))
        if token.lower() not in STOP_WORDS
    }


def token_dice(left: object, right: object) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return 2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))


def character_ngrams(value: object, size: int = 3) -> set[str]:
    normalized = re.sub(r"[^0-9a-z가-힣]", "", normalize_text(value).lower())
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    }


def character_dice(left: object, right: object) -> float:
    left_ngrams = character_ngrams(left)
    right_ngrams = character_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return 2 * len(left_ngrams & right_ngrams) / (len(left_ngrams) + len(right_ngrams))


def text_overlap(left: object, right: object) -> float:
    return 0.5 * token_dice(left, right) + 0.5 * character_dice(left, right)


def extract_conditions(value: object) -> set[str]:
    return {
        re.sub(r"\s+", "", match).lower()
        for match in CONDITION_PATTERN.findall(normalize_text(value))
    }


def condition_overlap(query: dict[str, str], candidate: dict[str, Any]) -> float:
    query_conditions = set()
    candidate_conditions = set()
    for field in ("target_audience", "selection_criteria", "benefits"):
        query_conditions.update(extract_conditions(query.get(field)))
        candidate_conditions.update(extract_conditions(candidate.get(field)))
    if not query_conditions:
        return 0.0
    return len(query_conditions & candidate_conditions) / len(query_conditions)


def field_overlap_scores(
    query: dict[str, str], candidate: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    scores = {}
    active_weight = 0.0
    weighted_score = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        if not normalize_text(query.get(field)):
            continue
        score = text_overlap(query.get(field), candidate.get(field))
        scores[field] = score
        weighted_score += score * weight
        active_weight += weight
    if active_weight == 0:
        return 0.0, scores
    return weighted_score / active_weight, scores


def _clip(value: object, limit: int = 1000) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class PolicySimilarityService:
    def __init__(
        self,
        *,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        collection: Any | None = None,
        embedder: Any | None = None,
        corpus: tuple[dict[str, Any], ...] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self._collection = collection
        self._embedder = embedder
        self._corpus = corpus or load_policy_corpus()
        self._corpus_by_id = {policy["service_id"]: policy for policy in self._corpus}
        self._manifest = manifest

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            manifest_path = self.index_dir / "manifest.json"
            if not manifest_path.exists():
                raise PolicyIndexUnavailableError(
                    f"정책 유사도 인덱스 manifest가 없습니다: {manifest_path}"
                )
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    @property
    def collection(self) -> Any:
        if self._collection is None:
            if not self.index_dir.exists():
                raise PolicyIndexUnavailableError(
                    f"정책 유사도 인덱스가 없습니다: {self.index_dir}"
                )
            import chromadb

            client = chromadb.PersistentClient(path=str(self.index_dir))
            collection_name = self.manifest.get(
                "collection_name", DEFAULT_COLLECTION_NAME
            )
            self._collection = client.get_collection(collection_name)
            expected_count = int(self.manifest.get("document_count", 0))
            if expected_count and self._collection.count() != expected_count:
                raise PolicyIndexUnavailableError(
                    "정책 유사도 인덱스 건수가 manifest와 일치하지 않습니다."
                )
        return self._collection

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.manifest["embedding_model"]
            device = os.getenv("POLICY_EMBEDDING_DEVICE", "cpu")
            self._embedder = SentenceTransformer(model_name, device=device)
            self._embedder.max_seq_length = int(
                self.manifest.get("max_sequence_length", 512)
            )
        return self._embedder

    def search(
        self,
        policy: dict[str, Any],
        *,
        top_k: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k는 1에서 10 사이여야 합니다.")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score는 0에서 1 사이여야 합니다.")

        started_at = time.perf_counter()
        query = build_direct_policy_query(policy)
        if not query["policy_name"] or not query["benefits"]:
            raise ValueError("유사 정책 검색에는 정책명과 혜택이 필요합니다.")

        reference_date = (
            normalize_date(as_of_date or query.get("effective_date"))
            or date.today().isoformat()
        )
        query_document = build_policy_search_document(query)
        query_embedding = self.embedder.encode(
            query_document,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        candidate_count = min(max(top_k * 10, 50), self.collection.count())
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            where={
                "registered_ordinal": {
                    "$lte": date_ordinal(reference_date, default=date.today())
                }
            },
            include=["distances", "metadatas"],
        )

        excluded_id = query.get("service_id")
        ranked = []
        for service_id, distance, metadata in zip(
            raw.get("ids", [[]])[0],
            raw.get("distances", [[]])[0],
            raw.get("metadatas", [[]])[0],
            strict=True,
        ):
            if service_id == excluded_id:
                continue
            candidate = self._corpus_by_id.get(service_id)
            if candidate is None:
                continue

            dense_score = max(0.0, min(1.0, 1.0 - float(distance)))
            lexical_score, field_scores = field_overlap_scores(query, candidate)
            numeric_score = condition_overlap(query, candidate)
            final_score = (
                0.65 * dense_score + 0.25 * lexical_score + 0.10 * numeric_score
            )
            if final_score < min_score:
                continue

            reasons = [
                {
                    "field": FIELD_LABELS[field],
                    "score": round(score * 100, 1),
                }
                for field, score in sorted(
                    field_scores.items(), key=lambda item: item[1], reverse=True
                )
                if score >= 0.10
            ][:3]
            ranked.append(
                {
                    "service_id": service_id,
                    "policy_name": candidate["policy_name"],
                    "organization": candidate["organization"],
                    "category": candidate["category"],
                    "registered_at": candidate["registered_at"],
                    "modified_at": candidate["modified_at"],
                    "similarity_score": round(final_score * 100, 1),
                    "dense_score": round(dense_score * 100, 1),
                    "match_reasons": reasons,
                    "target_audience": _clip(candidate["target_audience"]),
                    "benefits": _clip(candidate["benefits"]),
                    "application_period": _clip(candidate["application_period"], 500),
                    "application_method": _clip(candidate["application_method"], 500),
                    "source_url": candidate["source_url"],
                    "_metadata": metadata,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["similarity_score"],
                item["dense_score"],
                item["service_id"],
            ),
            reverse=True,
        )
        for item in ranked:
            item.pop("_metadata", None)

        return {
            "as_of_date": reference_date,
            "index_version": self.manifest.get("built_at"),
            "source_count": self.manifest.get("document_count", len(self._corpus)),
            "query_time_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "results": ranked[:top_k],
        }


@lru_cache(maxsize=1)
def get_policy_similarity_service() -> PolicySimilarityService:
    return PolicySimilarityService()


def find_similar_policies(
    policy: dict[str, Any],
    *,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    return get_policy_similarity_service().search(
        policy,
        top_k=top_k,
        min_score=min_score,
        as_of_date=as_of_date,
    )
