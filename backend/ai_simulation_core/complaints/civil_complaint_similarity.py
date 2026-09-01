"""Complaint-text similarity search for public-FAQ reference examples.

Reference eligibility is based on semantic similarity between the generated
complaint text and each public FAQ document, with a lightweight core-topic check
to prevent generic words such as "support" from connecting unrelated subjects.
Policy region, age, and qualification metadata never block a result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (
    FAQ_DATA_DIR,
    PROJECT_ROOT,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
    normalize_text,
)


DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "civil_complaints" / "current"
DEFAULT_COLLECTION_NAME = "civil_complaint_reference"
DEFAULT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MANIFEST_SCHEMA_VERSION = 1
ACTIVE_POINTER_FILENAME = "active.json"
ACTIVE_POINTER_SCHEMA_VERSION = 1
ACTIVE_RELOAD_STRATEGY = "detect_active_pointer_per_request"

# Eligibility uses the generated complaint's dense similarity plus a core-topic
# overlap check. Component score fields remain for response compatibility, but
# only complaint-text dense similarity changes the displayed score.
COMPLAINT_DENSE_FLOOR = 0.40
TOPIC_LEXICAL_FLOOR = 0.08
SCORE_EPSILON = 1e-7

MAX_TOP_K = 5
MAX_CANDIDATE_COUNT = 300

COMMON_WARNINGS = [
    (
        "현재 정책의 지역·연령·자격 조건은 확인하지 않고, "
        "민원 문구 유사도와 핵심 주제 일치만으로 제시된 사례입니다."
    ),
    "공개 FAQ의 유사 사례일 뿐 동일한 자격 판정이나 처리 결과를 보장하지 않습니다.",
    "검색 신뢰도는 최대 medium이며 시민 여론이나 민원 발생률 예측으로 사용할 수 없습니다.",
]
LEXICAL_FALLBACK_WARNING = (
    "양쪽에서 공통 핵심 분야를 확인하지 못해 문구 겹침을 보조 기준으로 사용한 "
    "낮은 신뢰도 결과입니다."
)

TOKEN_PATTERN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[가-힣A-Za-z]{2,}")
STOP_WORDS = {
    "관련",
    "경우",
    "대한",
    "또는",
    "문의",
    "서비스",
    "신청",
    "있는",
    "지원",
    "정책",
    "하는",
    "하여",
    "해당",
}

DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "housing": (
        "월세",
        "전세",
        "주거",
        "주택",
        "임대",
        "보증금",
        "전월세",
        "기숙사",
        "housing",
        "rent",
        "rental",
        "homeowner",
        "residence",
    ),
    "employment": (
        "취업",
        "고용",
        "일자리",
        "구직",
        "직업",
        "직업훈련",
        "내일배움카드",
        "채용",
        "진로",
        "창업",
        "근로",
        "employment",
        "job",
        "career",
        "startup",
    ),
    "education": (
        "교육",
        "학교",
        "대학",
        "학생",
        "학자금",
        "등록금",
        "장학",
        "보육",
        "education",
        "school",
        "university",
        "tuition",
        "scholarship",
    ),
    "health": (
        "건강",
        "의료",
        "병원",
        "보건소",
        "진료",
        "치료",
        "질병",
        "결핵",
        "치매",
        "금연클리닉",
        "예방접종",
        "health",
        "medical",
        "hospital",
        "treatment",
        "vaccination",
    ),
    "family_care": (
        "출산",
        "육아",
        "아동",
        "가족",
        "돌봄",
        "양육",
        "어린이집",
        "childbirth",
        "childcare",
        "family",
        "caregiving",
    ),
    "agriculture_fisheries": (
        "농업",
        "농어업",
        "어업",
        "축산",
        "임업",
        "산림",
        "산지전용",
        "산지관리",
        "수산물",
        "낚시",
        "어선",
        "귀농",
        "귀어",
        "agriculture",
        "farming",
        "fishery",
        "livestock",
        "forestry",
    ),
    "transport": (
        "교통",
        "버스",
        "철도",
        "자동차",
        "주차",
        "운전",
        "transport",
        "transit",
        "railway",
        "vehicle",
        "parking",
    ),
    "business_finance": (
        "사업자",
        "소상공",
        "중소기업",
        "대출",
        "융자",
        "금융",
        "채무",
        "business",
        "small business",
        "loan",
        "finance",
        "debt",
    ),
    "tax": (
        "세금",
        "과세",
        "납세",
        "지방세",
        "주민세",
        "소득세",
        "재산세",
        "tax",
        "taxation",
    ),
    "environment": (
        "환경",
        "폐기물",
        "쓰레기",
        "수질",
        "대기",
        "소음",
        "environment",
        "waste",
        "water quality",
        "air quality",
        "noise",
    ),
    "culture_sports": (
        "문화",
        "예술",
        "체육",
        "스포츠",
        "관광",
        "도서관",
        "culture",
        "cultural",
        "sports",
        "tourism",
        "library",
    ),
    "legal_administration": (
        "허가",
        "등록",
        "증명서",
        "여권",
        "주민등록",
        "행정처분",
        "permit",
        "license",
        "registration",
        "certificate",
        "passport",
        "administrative",
    ),
    "communications": (
        "휴대전화",
        "휴대폰",
        "이동통신",
        "통신요금",
        "전화요금",
        "알뜰폰",
        "유심",
        "번호이동",
        "전기통신",
        "telecommunication",
        "mobile phone",
    ),
    "security_crime": (
        "경찰",
        "범죄",
        "수사기관",
        "형사사건",
        "절도",
        "폭행",
        "성범죄",
        "스토킹",
        "고소장",
        "피의자",
        "police",
        "crime",
        "criminal",
    ),
    "disaster_safety": (
        "재난",
        "화재",
        "소방",
        "산불",
        "지진",
        "홍수",
        "침수",
        "대피소",
        "구급",
        "구조대",
        "disaster",
        "wildfire",
        "earthquake",
        "flood",
        "firefighter",
    ),
    "consumer_protection": (
        "소비자",
        "환불",
        "청약철회",
        "리콜",
        "분쟁조정",
        "피해구제",
        "전자상거래",
        "불공정약관",
        "consumer",
        "refund",
        "recall",
    ),
    "immigration": (
        "출입국",
        "외국인",
        "체류자격",
        "체류기간",
        "입국 비자",
        "취업비자",
        "유학비자",
        "비자 발급",
        "비자 연장",
        "귀화",
        "국적취득",
        "결혼이민",
        "immigration",
        "foreigner",
        "visa",
        "naturalization",
    ),
    "defense_veterans": (
        "군인",
        "군 장병",
        "장병 내일준비적금",
        "군복무",
        "병역",
        "입영",
        "예비군",
        "사회복무요원",
        "보훈",
        "국가유공자",
        "제대군인",
        "전역자",
        "현역병",
        "military",
        "veteran",
    ),
    "animal_welfare": (
        "반려동물",
        "유기동물",
        "동물등록",
        "동물보호소",
        "동물학대",
        "개물림",
        "맹견",
        "pet care",
        "animal welfare",
    ),
    "privacy_digital_rights": (
        "개인정보",
        "정보유출",
        "명의도용",
        "해킹",
        "사이버범죄",
        "디지털성범죄",
        "정보보호",
        "privacy",
        "personal data",
        "data breach",
        "identity theft",
    ),
}

DOMAIN_LABELS = {
    "housing": "주거·월세",
    "employment": "취업·고용",
    "education": "교육",
    "health": "건강·의료",
    "family_care": "가족·돌봄",
    "agriculture_fisheries": "농어업",
    "transport": "교통",
    "business_finance": "소상공·금융",
    "tax": "세금",
    "environment": "환경",
    "culture_sports": "문화·체육",
    "legal_administration": "행정·증명",
    "communications": "통신·휴대전화",
    "security_crime": "치안·범죄",
    "disaster_safety": "재난·안전",
    "consumer_protection": "소비자 보호",
    "immigration": "이민·외국인",
    "defense_veterans": "국방·보훈",
    "animal_welfare": "동물복지",
    "privacy_digital_rights": "개인정보·디지털 권리",
}


class CivilComplaintIndexUnavailableError(RuntimeError):
    """Raised when a verified public-FAQ index cannot be served."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_active_index_dir(
    logical_index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> tuple[Path, str | None, dict[str, Any] | None]:
    """Resolve an atomic active pointer, or return the legacy current directory.

    The pointer target must stay below ``civil_complaints/versions`` and bind
    the target manifest hash.  A malformed pointer fails closed instead of
    silently falling back to the legacy Chroma directory.
    """

    logical_dir = Path(logical_index_dir).resolve()
    pointer_path = logical_dir / ACTIVE_POINTER_FILENAME
    if not pointer_path.exists():
        return logical_dir, None, None
    if not pointer_path.is_file():
        raise CivilComplaintIndexUnavailableError(
            f"민원 FAQ active pointer가 일반 파일이 아닙니다: {pointer_path}"
        )

    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CivilComplaintIndexUnavailableError(
            f"민원 FAQ active pointer를 읽을 수 없습니다: {pointer_path}"
        ) from error
    if not isinstance(pointer, dict):
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer 형식이 올바르지 않습니다."
        )
    try:
        schema_version = int(pointer.get("schema_version"))
    except (TypeError, ValueError) as error:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer에 schema_version이 없습니다."
        ) from error
    if schema_version != ACTIVE_POINTER_SCHEMA_VERSION:
        raise CivilComplaintIndexUnavailableError(
            "지원하지 않는 민원 FAQ active pointer schema_version입니다."
        )
    if pointer.get("reload_strategy") != ACTIVE_RELOAD_STRATEGY:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer의 reload_strategy가 지원되지 않습니다."
        )

    version_path = normalize_text(pointer.get("version_path"))
    active_version = normalize_text(pointer.get("active_version"))
    expected_manifest_hash = normalize_text(pointer.get("manifest_sha256"))
    if not version_path or not active_version or not expected_manifest_hash:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer에 버전 경로 또는 manifest hash가 없습니다."
        )

    versions_root = (logical_dir.parent / "versions").resolve()
    resolved_dir = (logical_dir / version_path).resolve()
    try:
        resolved_dir.relative_to(versions_root)
    except ValueError as error:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer가 versions 디렉터리 밖을 가리킵니다."
        ) from error
    if resolved_dir == versions_root or resolved_dir.name != active_version:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active pointer의 버전 이름과 경로가 일치하지 않습니다."
        )
    if not resolved_dir.is_dir():
        raise CivilComplaintIndexUnavailableError(
            f"민원 FAQ active version 디렉터리가 없습니다: {resolved_dir}"
        )
    manifest_path = resolved_dir / "manifest.json"
    try:
        manifest_hash = _sha256_bytes(manifest_path.read_bytes())
    except OSError as error:
        raise CivilComplaintIndexUnavailableError(
            f"민원 FAQ active version manifest를 읽을 수 없습니다: {manifest_path}"
        ) from error
    if manifest_hash != expected_manifest_hash:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ active version manifest hash가 pointer와 다릅니다."
        )
    return resolved_dir, _sha256_bytes(pointer_bytes), pointer


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _tag_text(text: object, patterns: Mapping[str, Sequence[str]]) -> set[str]:
    normalized = normalize_text(text).lower()
    return {
        tag
        for tag, expressions in patterns.items()
        if _contains_any(normalized, expressions)
    }


def domain_tags(text: object) -> set[str]:
    return _tag_text(text, DOMAIN_PATTERNS)


def _tokens(value: object) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(normalize_text(value))
        if token.lower() not in STOP_WORDS
    }


def _dice(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def _character_ngrams(value: object, size: int = 3) -> set[str]:
    text = re.sub(r"[^0-9a-z가-힣]", "", normalize_text(value).lower())
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def lexical_score(left: object, right: object) -> float:
    return 0.6 * _dice(_tokens(left), _tokens(right)) + 0.4 * _dice(
        _character_ngrams(left), _character_ngrams(right)
    )


def _topic_overlap_evidence(
    query_text: object,
    candidate_heading: object,
    *,
    lexical: float,
) -> tuple[bool, dict[str, Any]]:
    query_domains = domain_tags(query_text)
    candidate_domains = domain_tags(candidate_heading)
    shared_domains = query_domains & candidate_domains

    if query_domains and candidate_domains:
        matched = bool(shared_domains)
        basis = "shared_core_topic" if matched else "conflicting_core_topics"
    else:
        matched = lexical + SCORE_EPSILON >= TOPIC_LEXICAL_FLOOR
        basis = "lexical_fallback" if matched else "no_core_topic_overlap"

    return matched, {
        "basis": basis,
        "query_domains": sorted(query_domains),
        "candidate_domains": sorted(candidate_domains),
        "shared_domains": sorted(shared_domains),
        "lexical_overlap": round(lexical * 100, 1),
    }


def _clip(value: object, limit: int = 1600) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _empty_rejection_counts() -> dict[str, int]:
    return {"topic": 0, "below_complaint_dense": 0}


def _embedding_matrix(embedder: Any, documents: list[str]) -> np.ndarray:
    encoded = embedder.encode(
        documents,
        batch_size=min(64, max(1, len(documents))),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(documents) or matrix.shape[1] == 0:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ 검색 임베딩의 형태가 올바르지 않습니다."
        )
    return matrix


def _distance_scores(raw: Mapping[str, Any], row_index: int) -> dict[str, float]:
    ids_rows = raw.get("ids")
    distance_rows = raw.get("distances")
    if not isinstance(ids_rows, list) or not isinstance(distance_rows, list):
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ Chroma 응답에 ids 또는 distances가 없습니다."
        )
    try:
        ids = ids_rows[row_index]
        distances = distance_rows[row_index]
    except IndexError as error:
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ Chroma 배치 응답 건수가 요청과 다릅니다."
        ) from error
    if len(ids) != len(distances):
        raise CivilComplaintIndexUnavailableError(
            "민원 FAQ Chroma 후보 ID와 거리 건수가 다릅니다."
        )
    scores = {}
    for case_id, distance in zip(ids, distances, strict=True):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        scores[normalize_text(case_id)] = score
    return scores


def _text_similarity_match_reasons(
    score: float,
    topic_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons = [
        {
            "type": "semantic",
            "label": "민원 문구·내용 유사",
            "details": f"유사도 {score * 100:.1f}%",
        }
    ]
    shared_domains = topic_evidence.get("shared_domains")
    if isinstance(shared_domains, list) and shared_domains:
        labels = [DOMAIN_LABELS.get(value, value) for value in shared_domains]
        reasons.append(
            {
                "type": "topic",
                "label": "핵심 주제 일치",
                "details": "·".join(labels),
            }
        )
    else:
        reasons.append(
            {
                "type": "topic",
                "label": "핵심 표현 일치",
                "details": (
                    f"문구 겹침 {float(topic_evidence.get('lexical_overlap', 0.0)):.1f}%"
                ),
            }
        )
    return reasons


def _score_payload(
    *,
    complaint_dense: float,
    policy_dense: float,
    semantic: float,
    lexical: float,
    context: float,
    final: float,
) -> dict[str, float]:
    return {
        "complaint_dense": round(complaint_dense * 100, 1),
        "policy_dense": round(policy_dense * 100, 1),
        "semantic": round(semantic * 100, 1),
        "lexical": round(lexical * 100, 1),
        "context": round(context * 100, 1),
        "final": round(final * 100, 1),
    }


class CivilComplaintSimilarityService:
    def __init__(
        self,
        *,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        data_dir: str | Path = FAQ_DATA_DIR,
        collection: Any | None = None,
        embedder: Any | None = None,
        corpus: Sequence[dict[str, Any]] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self._collection = collection
        self._client: Any | None = None
        self._embedder = embedder
        self._corpus = tuple(corpus) if corpus is not None else None
        self._corpus_by_id: dict[str, dict[str, Any]] | None = None
        self._manifest = manifest
        self._source_verified = False
        self._collection_verified = False
        self._active_index_dir: Path | None = None
        self._active_pointer_digest: str | None = None
        self._active_state_initialized = False
        self._active_pointer: dict[str, Any] | None = None
        self._state_lock = threading.RLock()

    def _refresh_active_index(self) -> None:
        """Detect pointer swaps and discard every version-bound cache."""

        with self._state_lock:
            resolved_dir, pointer_digest, pointer = resolve_active_index_dir(
                self.index_dir
            )
            if not self._active_state_initialized:
                self._active_index_dir = resolved_dir
                self._active_pointer_digest = pointer_digest
                self._active_pointer = pointer
                self._active_state_initialized = True
                return
            if (
                resolved_dir == self._active_index_dir
                and pointer_digest == self._active_pointer_digest
            ):
                return

            self._active_index_dir = resolved_dir
            self._active_pointer_digest = pointer_digest
            self._active_pointer = pointer
            self._manifest = None
            self._collection = None
            self._client = None
            self._embedder = None
            self._corpus = None
            self._corpus_by_id = None
            self._source_verified = False
            self._collection_verified = False

    @property
    def active_index_dir(self) -> Path:
        self._refresh_active_index()
        assert self._active_index_dir is not None
        return self._active_index_dir

    @property
    def manifest(self) -> dict[str, Any]:
        self._refresh_active_index()
        if self._manifest is None:
            assert self._active_index_dir is not None
            path = self._active_index_dir / "manifest.json"
            if not path.is_file():
                raise CivilComplaintIndexUnavailableError(
                    f"민원 FAQ 인덱스 manifest가 없습니다: {path}"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CivilComplaintIndexUnavailableError(
                    f"민원 FAQ 인덱스 manifest를 읽을 수 없습니다: {path}"
                ) from error
            if not isinstance(payload, dict):
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ 인덱스 manifest 형식이 올바르지 않습니다."
                )
            self._manifest = payload

        try:
            schema_version = int(self._manifest.get("schema_version"))
        except (TypeError, ValueError) as error:
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ manifest에 schema_version이 없습니다."
            ) from error
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise CivilComplaintIndexUnavailableError(
                "지원하지 않는 민원 FAQ manifest schema_version입니다."
            )
        for key in ("collection_name", "embedding_model", "built_at"):
            if not normalize_text(self._manifest.get(key)):
                raise CivilComplaintIndexUnavailableError(
                    f"민원 FAQ manifest에 {key}가 없습니다."
                )
        source = self._manifest.get("source")
        if (
            not isinstance(source, dict)
            or source.get("source_kind") != "public_faq_snapshot"
            or source.get("matched_policy_used") is not False
        ):
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ manifest는 matchedPolicy를 배제한 public_faq_snapshot "
                "출처여야 합니다."
            )
        if self._active_pointer is not None:
            activation = self._manifest.get("activation")
            if (
                not isinstance(activation, dict)
                or activation.get("kind") != "atomic_pointer"
                or activation.get("reload_strategy") != ACTIVE_RELOAD_STRATEGY
            ):
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ active version manifest의 activation 계약이 "
                    "올바르지 않습니다."
                )
        return self._manifest

    def _verify_source(self) -> None:
        self._refresh_active_index()
        if self._source_verified:
            return
        try:
            current = civil_complaint_source_fingerprint(self.data_dir)
        except (OSError, ValueError) as error:
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ 정본 스냅샷을 검증할 수 없습니다."
            ) from error

        expected = {
            key: self.manifest.get(key)
            for key in (
                "detail_sha256",
                "metadata_sha256",
                "raw_record_count",
                "unique_count",
            )
        }
        if expected != current:
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ 정본 hash 또는 건수가 manifest와 다릅니다. "
                "scripts/rag/build_civil_complaint_index.py로 재구축하세요."
            )
        self._source_verified = True

    @property
    def corpus(self) -> tuple[dict[str, Any], ...]:
        self._refresh_active_index()
        self._verify_source()
        if self._corpus is None:
            try:
                self._corpus = load_civil_complaint_corpus(self.data_dir)
            except (OSError, ValueError) as error:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ 정본 코퍼스를 읽을 수 없습니다."
                ) from error
        if len(self._corpus) != int(self.manifest["unique_count"]):
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ 정본 코퍼스 건수가 manifest와 다릅니다."
            )
        return self._corpus

    @property
    def corpus_by_id(self) -> dict[str, dict[str, Any]]:
        if self._corpus_by_id is None:
            self._corpus_by_id = {
                normalize_text(record.get("case_id")): record for record in self.corpus
            }
            if "" in self._corpus_by_id or len(self._corpus_by_id) != len(self.corpus):
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ canonical case_id가 비어 있거나 중복입니다."
                )
        return self._corpus_by_id

    @property
    def collection(self) -> Any:
        self._refresh_active_index()
        self._verify_source()
        if self._collection is None:
            assert self._active_index_dir is not None
            if not self._active_index_dir.is_dir():
                raise CivilComplaintIndexUnavailableError(
                    f"민원 FAQ 인덱스가 없습니다: {self._active_index_dir}"
                )
            try:
                import chromadb

                self._client = chromadb.PersistentClient(
                    path=str(self._active_index_dir)
                )
                self._collection = self._client.get_collection(
                    normalize_text(self.manifest["collection_name"])
                )
            except Exception as error:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma collection을 열 수 없습니다."
                ) from error
        if not self._collection_verified:
            try:
                count = int(self._collection.count())
            except Exception as error:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma collection 건수를 확인할 수 없습니다."
                ) from error
            if count != int(self.manifest["unique_count"]):
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma 건수가 manifest와 다릅니다."
                )
            self._collection_verified = True
        return self._collection

    @property
    def embedder(self) -> Any:
        self._refresh_active_index()
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                device = os.getenv("CIVIL_COMPLAINT_EMBEDDING_DEVICE", "cpu")
                self._embedder = SentenceTransformer(
                    normalize_text(self.manifest["embedding_model"]),
                    device=device,
                )
                self._embedder.max_seq_length = int(
                    self.manifest.get("max_sequence_length", 512)
                )
            except Exception as error:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ 임베딩 모델을 불러올 수 없습니다."
                ) from error
        return self._embedder

    @property
    def source_metadata(self) -> dict[str, Any]:
        return dict(self.manifest["source"])

    def _base_response(self) -> dict[str, Any]:
        return {
            "index_version": self.manifest["built_at"],
            "source_count": int(self.manifest["unique_count"]),
            "source": self.source_metadata,
            "rejection_counts": _empty_rejection_counts(),
            "warnings": list(COMMON_WARNINGS),
        }

    def _prepare_query(self, item: object) -> tuple[dict[str, Any] | None, list[str]]:
        if not isinstance(item, Mapping):
            return None, ["item_must_be_object"]
        complaint_text = normalize_text(item.get("complaint_text"))
        if not complaint_text:
            return None, ["complaint_text_required"]
        return {"complaint_text": complaint_text}, []

    def search_batch(
        self,
        items: list[dict],
        *,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise ValueError("items는 list여야 합니다.")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k는 1에서 {MAX_TOP_K} 사이여야 합니다.")
        if not items:
            return []

        started_at = time.perf_counter()
        # Verify source, manifest, collection count, and canonical IDs before any
        # item can be labelled invalid or matched.
        self._refresh_active_index()
        active_generation = (
            self._active_index_dir,
            self._active_pointer_digest,
        )
        collection = self.collection
        corpus_by_id = self.corpus_by_id
        if active_generation != (
            self._active_index_dir,
            self._active_pointer_digest,
        ):
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ active version이 검색 준비 중 변경되었습니다. "
                "요청을 다시 시도하세요."
            )
        prepared: list[dict[str, Any] | None] = []
        responses: list[dict[str, Any] | None] = []
        valid_positions: list[int] = []
        base = self._base_response()
        for position, item in enumerate(items):
            query, errors = self._prepare_query(item)
            prepared.append(query)
            if query is None:
                responses.append(
                    {
                        **base,
                        "status": "invalid_query",
                        "results": [],
                        "validation_errors": errors,
                        "rejection_counts": _empty_rejection_counts(),
                    }
                )
            else:
                responses.append(None)
                valid_positions.append(position)

        if valid_positions:
            complaint_documents = [
                prepared[position]["complaint_text"]  # type: ignore[index]
                for position in valid_positions
            ]
            embeddings = _embedding_matrix(self.embedder, complaint_documents)
            collection_count = int(collection.count())
            candidate_count = min(MAX_CANDIDATE_COUNT, collection_count)
            if candidate_count < 1:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma collection이 비어 있습니다."
                )
            try:
                raw = collection.query(
                    query_embeddings=embeddings.tolist(),
                    n_results=candidate_count,
                    include=["distances"],
                )
            except Exception as error:
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma 배치 검색에 실패했습니다."
                ) from error
            if not isinstance(raw, Mapping):
                raise CivilComplaintIndexUnavailableError(
                    "민원 FAQ Chroma 배치 응답 형식이 올바르지 않습니다."
                )

            for batch_index, position in enumerate(valid_positions):
                query = prepared[position]
                assert query is not None
                complaint_scores = _distance_scores(raw, batch_index)
                rejection_counts = _empty_rejection_counts()
                ranked: list[tuple[float, float, str, dict[str, Any]]] = []

                for case_id, complaint_dense in complaint_scores.items():
                    record = corpus_by_id.get(case_id)
                    if record is None:
                        raise CivilComplaintIndexUnavailableError(
                            f"민원 FAQ Chroma ID가 canonical 정본에 없습니다: {case_id}"
                        )

                    candidate_heading = f"{record['title']} {record['question']}"
                    lexical = lexical_score(query["complaint_text"], candidate_heading)
                    if complaint_dense + SCORE_EPSILON < COMPLAINT_DENSE_FLOOR:
                        rejection_counts["below_complaint_dense"] += 1
                        continue
                    topic_matches, topic_evidence = _topic_overlap_evidence(
                        query["complaint_text"],
                        candidate_heading,
                        lexical=lexical,
                    )
                    if not topic_matches:
                        rejection_counts["topic"] += 1
                        continue

                    # Keep the component-score shape stable for existing clients,
                    # while making every eligibility score equal to the one dense
                    # complaint-text similarity. Lexical overlap is tie-break only.
                    policy_dense = 0.0
                    semantic = complaint_dense
                    context = 0.0
                    final = complaint_dense
                    evidence = {
                        "matching_basis": "complaint_text_similarity",
                        "complaint_similarity": round(complaint_dense * 100, 1),
                        "topic_overlap": topic_evidence,
                    }

                    score_payload = _score_payload(
                        complaint_dense=complaint_dense,
                        policy_dense=policy_dense,
                        semantic=semantic,
                        lexical=lexical,
                        context=context,
                        final=final,
                    )
                    used_lexical_fallback = (
                        topic_evidence["basis"] == "lexical_fallback"
                    )
                    result_warnings = list(COMMON_WARNINGS)
                    if used_lexical_fallback:
                        result_warnings.append(LEXICAL_FALLBACK_WARNING)
                    result = {
                        "case_id": record["case_id"],
                        "title": record["title"],
                        "question": _clip(record["question"]),
                        "answer": _clip(record["answer"]),
                        "organization": record["organization"],
                        "related_laws": record["related_laws"],
                        "source_kind": "public_faq_snapshot",
                        "reference_eligible": True,
                        "match_score": score_payload["final"],
                        "component_scores": score_payload,
                        "confidence": (
                            "low"
                            if used_lexical_fallback
                            else "medium" if final >= 0.78 else "low"
                        ),
                        "match_reasons": _text_similarity_match_reasons(
                            complaint_dense,
                            topic_evidence,
                        ),
                        "evidence": evidence,
                        "warnings": result_warnings,
                    }
                    ranked.append((complaint_dense, lexical, case_id, result))

                ranked.sort(
                    key=lambda item: (item[0], item[1], item[2]),
                    reverse=True,
                )
                results = [item[3] for item in ranked[:top_k]]
                responses[position] = {
                    **base,
                    "status": "matched" if results else "no_reliable_match",
                    "results": results,
                    "rejection_counts": rejection_counts,
                }

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        self._refresh_active_index()
        if active_generation != (
            self._active_index_dir,
            self._active_pointer_digest,
        ):
            raise CivilComplaintIndexUnavailableError(
                "민원 FAQ active version이 검색 중 변경되었습니다. 요청을 다시 시도하세요."
            )
        final_responses: list[dict[str, Any]] = []
        for response in responses:
            assert response is not None
            final_responses.append({**response, "query_time_ms": elapsed_ms})
        return final_responses


@lru_cache(maxsize=1)
def get_civil_complaint_similarity_service() -> CivilComplaintSimilarityService:
    return CivilComplaintSimilarityService()


def find_similar_complaint_cases_batch(
    items: list[dict],
    *,
    top_k: int = 1,
) -> list[dict]:
    """Search generated complaints in one encode/query batch."""

    if not isinstance(items, list):
        raise ValueError("items는 list여야 합니다.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k는 1에서 {MAX_TOP_K} 사이여야 합니다.")
    if not items:
        return []
    return get_civil_complaint_similarity_service().search_batch(
        items,
        top_k=top_k,
    )


__all__ = [
    "CivilComplaintIndexUnavailableError",
    "CivilComplaintSimilarityService",
    "COMPLAINT_DENSE_FLOOR",
    "ACTIVE_POINTER_FILENAME",
    "ACTIVE_POINTER_SCHEMA_VERSION",
    "ACTIVE_RELOAD_STRATEGY",
    "find_similar_complaint_cases_batch",
    "get_civil_complaint_similarity_service",
    "resolve_active_index_dir",
]
