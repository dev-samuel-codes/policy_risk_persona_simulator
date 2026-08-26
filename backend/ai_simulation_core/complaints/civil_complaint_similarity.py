"""Strict, provenance-bound reference search for generated complaints.

Dense similarity only retrieves candidates.  A candidate is never exposed until
region, domain, intent, exclusive qualification, age relationship, and all
shared score floors pass.  This prevents a superficially similar public FAQ
from being presented as evidence for a different policy or population.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (
    FAQ_DATA_DIR,
    PROJECT_ROOT,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
    load_civil_complaint_source_metadata,
    normalize_text,
)


DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "civil_complaints" / "current"
DEFAULT_COLLECTION_NAME = "civil_complaint_reference"
DEFAULT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MANIFEST_SCHEMA_VERSION = 1
ACTIVE_POINTER_FILENAME = "active.json"
ACTIVE_POINTER_SCHEMA_VERSION = 1
ACTIVE_RELOAD_STRATEGY = "detect_active_pointer_per_request"

# Retrieval and UI share these exact boundaries.  Do not duplicate literals in
# callers: boundary behavior is covered by dedicated tests.
COMPLAINT_DENSE_FLOOR = 0.50
POLICY_DENSE_FLOOR = 0.50
SEMANTIC_FLOOR = 0.55
FINAL_SCORE_FLOOR = 0.62
UI_REFERENCE_SCORE_FLOOR = 0.68
SCORE_EPSILON = 1e-7

MAX_TOP_K = 5
MIN_CANDIDATE_COUNT = 100
MAX_CANDIDATE_COUNT = 300
CANDIDATE_MULTIPLIER = 50

COMMON_WARNINGS = [
    "공개 FAQ의 유사 사례일 뿐 동일한 자격 판정이나 처리 결과를 보장하지 않습니다.",
    "검색 신뢰도는 최대 medium이며 시민 여론이나 민원 발생률 예측으로 사용할 수 없습니다.",
]

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
        "진료",
        "치료",
        "질병",
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
}

ISSUE_PATTERNS: dict[str, tuple[str, ...]] = {
    "eligibility": (
        "자격",
        "대상자",
        "지원대상",
        "해당되",
        "선정기준",
        "선정 조건",
        "지원 조건",
        "요건",
        "제외",
        "받을 수",
        "가능 여부",
        "왜 안",
    ),
    "documents": (
        "구비서류",
        "신청서류",
        "제출서류",
        "필요한 서류",
        "필요 서류",
        "증빙서류",
        "증명서류",
    ),
    "application_method": (
        "신청방법",
        "신청 방법",
        "신청절차",
        "신청 절차",
        "접수방법",
        "접수 방법",
        "어떻게 신청",
    ),
    "deadline": (
        "신청기간",
        "신청 기간",
        "접수기간",
        "접수 기간",
        "신청기한",
        "마감",
        "언제까지",
    ),
    "payment": (
        "지급액",
        "지원액",
        "지원금액",
        "지급일",
        "입금",
        "얼마",
        "수령",
    ),
    "appeal": ("이의신청", "불복", "재심", "반려", "처분 취소"),
    "report": ("신고방법", "신고 방법", "민원 접수", "불편 신고"),
}

AGE_ISSUE_PATTERN = re.compile(
    r"(?:나이|연령|만\s*\d{1,3}\s*세|\d{1,3}\s*세\s*(?:이상|이하|미만|초과))"
)
POLICY_AGE_CONTEXT_PATTERN = re.compile(
    r"(?:나이|연령|청년|청소년|아동|노인|고령|만\s*\d{1,3}\s*세|\d{1,3}\s*세\s*(?:이상|이하|미만|초과))"
)

QUALIFICATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "veteran": ("제대군인", "보훈대상", "국가유공자", "상이군경"),
    "disabled": ("장애인", "장애등록", "중증장애", "장애 정도"),
    "farmer_fisher": (
        "농어업인",
        "농업인",
        "어업인",
        "축산업자",
        "임업인",
    ),
    "single_parent": ("한부모", "미혼모", "미혼부"),
    "multicultural": ("다문화가족", "결혼이민자"),
    "basic_livelihood": ("기초생활수급", "수급권자", "차상위"),
    "small_business": ("소상공인", "자영업자"),
    "student": ("대학생", "재학생"),
    "newlywed": ("신혼부부",),
    "pregnant": ("임산부", "임신부"),
}

PROVINCE_ALIASES: dict[str, tuple[str, ...]] = {
    "서울": ("서울특별시", "서울시", "서울"),
    "부산": ("부산광역시", "부산시", "부산"),
    "대구": ("대구광역시", "대구시", "대구"),
    "인천": ("인천광역시", "인천시", "인천"),
    "광주": ("광주광역시", "광주시", "광주"),
    "대전": ("대전광역시", "대전시", "대전"),
    "울산": ("울산광역시", "울산시", "울산"),
    "세종": ("세종특별자치시", "세종시", "세종"),
    "경기": ("경기도", "경기"),
    "강원": ("강원특별자치도", "강원도", "강원"),
    "충북": ("충청북도", "충북"),
    "충남": ("충청남도", "충남"),
    "전북": ("전북특별자치도", "전라북도", "전북"),
    "전남": ("전라남도", "전남"),
    "경북": ("경상북도", "경북"),
    "경남": ("경상남도", "경남"),
    "제주": ("제주특별자치도", "제주도", "제주"),
}
CENTRAL_ORGANIZATION_PATTERN = re.compile(
    r"(?:^|\s)(?:[가-힣]+부|[가-힣]+처|[가-힣]+청|[가-힣]+위원회|국립[가-힣]+|[가-힣]+공단|[가-힣]+공사)(?:\s|$)"
)

AGE_RANGE_PATTERNS = (
    re.compile(
        r"(?:만\s*)?(\d{1,3})\s*세\s*(?:이상|부터)\s*(?:부터\s*)?.{0,24}?"
        r"(?:만\s*)?(\d{1,3})\s*세\s*(?:이하|까지)"
    ),
    re.compile(
        r"(?:만\s*)?(\d{1,3})\s*(?:세\s*)?(?:~|∼|～|－|-)\s*"
        r"(?:만\s*)?(\d{1,3})\s*세"
    ),
    re.compile(
        r"(?:만\s*)?(\d{1,3})\s*세\s*부터\s*(?:만\s*)?"
        r"(\d{1,3})\s*세\s*까지"
    ),
)
LOWER_AGE_PATTERN = re.compile(r"(?:만\s*)?(\d{1,3})\s*세\s*(이상|부터|초과)")
UPPER_AGE_PATTERN = re.compile(r"(?:만\s*)?(\d{1,3})\s*세\s*(이하|까지|미만)")


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


@dataclass(frozen=True)
class AgeRange:
    minimum: int | None
    maximum: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {"minimum": self.minimum, "maximum": self.maximum}


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


def issue_tags(text: object) -> set[str]:
    normalized = normalize_text(text).lower()
    tags = _tag_text(normalized, ISSUE_PATTERNS)
    if AGE_ISSUE_PATTERN.search(normalized):
        tags.update({"age", "eligibility"})
    return tags


def qualification_tags(text: object) -> set[str]:
    return _tag_text(text, QUALIFICATION_PATTERNS)


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


def _clip(value: object, limit: int = 1600) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _policy_field(policy: Mapping[str, Any], *names: str) -> str:
    detail = policy.get("상세정보")
    detail = detail if isinstance(detail, Mapping) else {}
    listing = policy.get("목록정보")
    listing = listing if isinstance(listing, Mapping) else {}
    for name in names:
        for container in (policy, detail, listing):
            value = normalize_text(container.get(name))
            if value:
                return value
    return ""


def build_policy_context_document(policy: Mapping[str, Any]) -> str:
    fields = (
        ("정책명", ("policy_name", "서비스명")),
        ("지원대상", ("target_audience", "지원대상")),
        ("지원내용", ("benefits", "지원내용")),
        ("선정기준", ("selection_criteria", "선정기준", "제외조건")),
        ("분야", ("category", "서비스분야")),
        ("지원유형", ("support_type", "지원유형")),
    )
    lines = []
    for label, names in fields:
        value = _policy_field(policy, *names)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _persona_document(persona: Mapping[str, Any]) -> str:
    fields = (
        "occupation",
        "persona",
        "professional_persona",
        "family_persona",
        "selection_cohort",
    )
    return " ".join(normalize_text(persona.get(key)) for key in fields)


def _canonical_province(value: object) -> str:
    province, _ = _matched_province_alias(value)
    return province


def _matched_province_alias(value: object) -> tuple[str, str]:
    text = normalize_text(value).replace("-", " ")
    matches: list[tuple[int, str, str]] = []
    for province, aliases in PROVINCE_ALIASES.items():
        for alias in aliases:
            index = text.find(alias)
            if index >= 0:
                matches.append((index, province, alias))
    if not matches:
        return "", ""
    _, province, alias = min(matches, key=lambda item: (item[0], -len(item[2])))
    return province, alias


def _normalize_district(value: object, province: str = "") -> str:
    text = normalize_text(value).replace("-", " ")
    if province:
        for alias in PROVINCE_ALIASES.get(province, ()):
            text = text.replace(alias, " ")
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def organization_region(organization: object) -> dict[str, str]:
    text = normalize_text(organization)
    province, alias = _matched_province_alias(text)
    if province:
        alias_end = text.find(alias) + len(alias)
        remainder = normalize_text(text[alias_end:])
        district = _normalize_district(remainder, province)
        return {
            "kind": "local",
            "province": province,
            "district": district,
            "raw": text,
        }
    if CENTRAL_ORGANIZATION_PATTERN.search(f" {text} "):
        return {"kind": "central", "province": "", "district": "", "raw": text}
    return {"kind": "unknown", "province": "", "district": "", "raw": text}


def _policy_region(policy: Mapping[str, Any]) -> tuple[dict[str, str] | None, str]:
    scope = normalize_text(policy.get("region_scope")).lower()
    province_text = normalize_text(
        policy.get("region_province") or policy.get("province")
    )
    district_text = normalize_text(
        policy.get("region_district") or policy.get("district")
    )
    if not scope and province_text:
        scope = "specific"
    if scope not in {"nationwide", "specific"}:
        return None, "policy_region_scope_unknown"
    if scope == "nationwide":
        if province_text or district_text:
            return None, "nationwide_policy_has_local_region"
        return {"scope": scope, "province": "", "district": ""}, ""

    province = _canonical_province(province_text)
    if not province:
        return None, "policy_province_unknown"
    district = _normalize_district(district_text, province)
    return {"scope": scope, "province": province, "district": district}, ""


def extract_age_ranges(value: object) -> tuple[AgeRange, ...]:
    text = normalize_text(value)
    ranges: set[AgeRange] = set()
    for pattern in AGE_RANGE_PATTERNS:
        for match in pattern.finditer(text):
            minimum, maximum = (int(part) for part in match.groups())
            if 0 <= minimum <= maximum <= 120:
                ranges.add(AgeRange(minimum, maximum))
    if ranges:
        return tuple(
            sorted(ranges, key=lambda item: (item.minimum or -1, item.maximum or 999))
        )

    lowers = [
        int(value) + (1 if comparator == "초과" else 0)
        for value, comparator in LOWER_AGE_PATTERN.findall(text)
    ]
    uppers = [
        int(value) - (1 if comparator == "미만" else 0)
        for value, comparator in UPPER_AGE_PATTERN.findall(text)
    ]
    if lowers and uppers:
        for minimum in lowers:
            for maximum in uppers:
                if 0 <= minimum <= maximum <= 120:
                    ranges.add(AgeRange(minimum, maximum))
    elif lowers:
        ranges.update(AgeRange(value, None) for value in lowers if 0 <= value <= 120)
    elif uppers:
        ranges.update(AgeRange(None, value) for value in uppers if 0 <= value <= 120)
    return tuple(
        sorted(
            ranges,
            key=lambda item: (
                -1 if item.minimum is None else item.minimum,
                999 if item.maximum is None else item.maximum,
            ),
        )
    )


def _optional_age(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        age = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return age if 0 <= age <= 120 else None


def _policy_age_range(policy: Mapping[str, Any], policy_text: str) -> AgeRange | None:
    minimum = _optional_age(policy.get("age_min"))
    maximum = _optional_age(policy.get("age_max"))
    if minimum is not None or maximum is not None:
        if minimum is not None and maximum is not None and minimum > maximum:
            return None
        return AgeRange(minimum, maximum)
    ranges = extract_age_ranges(policy_text)
    return ranges[0] if len(ranges) == 1 else None


def _persona_age_relationship(age: int, expected: AgeRange) -> str:
    lower_ok = expected.minimum is None or age >= expected.minimum
    upper_ok = expected.maximum is None or age <= expected.maximum
    if lower_ok and upper_ok:
        return "eligible"
    if expected.minimum is not None and age == expected.minimum - 1:
        return "lower_boundary"
    if expected.maximum is not None and age == expected.maximum + 1:
        return "upper_boundary"
    return "outside"


def _ranges_compatible(candidate: AgeRange, expected: AgeRange) -> bool:
    if expected.minimum is not None and candidate.minimum != expected.minimum:
        return False
    if expected.maximum is not None and candidate.maximum != expected.maximum:
        return False
    return True


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _region_gate(
    query_region: Mapping[str, str], candidate_region: Mapping[str, str]
) -> tuple[bool, dict[str, Any]]:
    evidence = {
        "policy_scope": query_region["scope"],
        "policy_province": query_region["province"],
        "policy_district": query_region["district"],
        "candidate_kind": candidate_region["kind"],
        "candidate_province": candidate_region["province"],
        "candidate_district": candidate_region["district"],
    }
    if query_region["scope"] == "nationwide":
        return True, {**evidence, "reason": "nationwide_policy"}
    if candidate_region["kind"] != "local":
        return False, {**evidence, "reason": "candidate_region_unknown_or_central"}
    if candidate_region["province"] != query_region["province"]:
        return False, {**evidence, "reason": "province_mismatch"}
    district = query_region["district"]
    if district and candidate_region["district"] != district:
        return False, {**evidence, "reason": "district_mismatch_or_unknown"}
    return True, {**evidence, "reason": "same_applicable_region"}


def _age_gate(
    *,
    active: bool,
    expected: AgeRange | None,
    persona: Mapping[str, Any],
    candidate_answer: str,
) -> tuple[bool, dict[str, Any]]:
    if not active:
        return True, {"active": False, "reason": "policy_has_no_age_issue"}
    if expected is None:
        return False, {"active": True, "reason": "policy_age_range_unknown"}
    persona_age = _optional_age(persona.get("age"))
    if persona_age is None:
        return False, {
            "active": True,
            "policy_range": expected.as_dict(),
            "reason": "persona_age_unknown",
        }
    relationship = _persona_age_relationship(persona_age, expected)
    candidate_ranges = extract_age_ranges(candidate_answer)
    compatible = next(
        (item for item in candidate_ranges if _ranges_compatible(item, expected)),
        None,
    )
    evidence = {
        "active": True,
        "policy_range": expected.as_dict(),
        "persona_age": persona_age,
        "persona_relationship": relationship,
        "candidate_ranges": [item.as_dict() for item in candidate_ranges],
        "matched_candidate_range": compatible.as_dict() if compatible else None,
    }
    if relationship == "outside":
        return False, {**evidence, "reason": "persona_outside_policy_or_boundary"}
    if not candidate_ranges:
        return False, {**evidence, "reason": "candidate_age_range_unknown"}
    if compatible is None:
        return False, {**evidence, "reason": "candidate_age_range_mismatch"}
    return True, {**evidence, "reason": "same_age_rule_relationship"}


def _empty_rejection_counts() -> dict[str, int]:
    return {
        "region": 0,
        "domain": 0,
        "issue": 0,
        "qualification": 0,
        "age": 0,
        "missing_dense_score": 0,
        "below_complaint_dense": 0,
        "below_policy_dense": 0,
        "below_semantic": 0,
        "below_final": 0,
        "below_ui_threshold": 0,
    }


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


def _candidate_tags(record: Mapping[str, Any]) -> dict[str, set[str]]:
    heading = f"{normalize_text(record.get('title'))} {normalize_text(record.get('question'))}"
    full_text = f"{heading} {normalize_text(record.get('answer'))}"
    return {
        "domains": domain_tags(heading),
        "issues": issue_tags(heading),
        "qualifications": qualification_tags(full_text),
    }


def _hard_gate_evidence(
    query: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any], float]:
    tags = _candidate_tags(record)
    candidate_region = organization_region(record.get("organization"))
    region_ok, region_evidence = _region_gate(query["region"], candidate_region)
    domain_matches = set(query["domains"]) & tags["domains"]
    issue_matches = set(query["issues"]) & tags["issues"]
    candidate_only_qualifications = tags["qualifications"] - set(
        query["qualifications"]
    )
    age_ok, age_evidence = _age_gate(
        active=bool(query["age_active"]),
        expected=query["age_range"],
        persona=query["persona"],
        candidate_answer=normalize_text(record.get("answer")),
    )

    failures = []
    if not region_ok:
        failures.append("region")
    if not query["domains"] or not tags["domains"] or not domain_matches:
        failures.append("domain")
    if not query["issues"] or not tags["issues"] or not issue_matches:
        failures.append("issue")
    if candidate_only_qualifications:
        failures.append("qualification")
    if not age_ok:
        failures.append("age")

    evidence = {
        "region": region_evidence,
        "domain_tags": sorted(domain_matches),
        "query_domain_tags": sorted(query["domains"]),
        "candidate_domain_tags": sorted(tags["domains"]),
        "issue_tags": sorted(issue_matches),
        "query_issue_tags": sorted(query["issues"]),
        "candidate_issue_tags": sorted(tags["issues"]),
        "qualification_tags": sorted(tags["qualifications"]),
        "candidate_only_qualification_tags": sorted(candidate_only_qualifications),
        "age": age_evidence,
    }
    context_parts = [
        1.0 if region_ok else 0.0,
        _set_overlap(set(query["domains"]), tags["domains"]),
        _set_overlap(set(query["issues"]), tags["issues"]),
        1.0 if not candidate_only_qualifications else 0.0,
        1.0 if age_ok else 0.0,
    ]
    return failures, evidence, sum(context_parts) / len(context_parts)


def _match_reasons(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    region = evidence["region"]
    if region["policy_scope"] == "nationwide":
        region_details: Any = "전국 정책"
    else:
        region_details = {
            "province": region["policy_province"],
            "district": region["policy_district"],
        }
    reasons.append(
        {"type": "region", "label": "정책 적용 지역 일치", "details": region_details}
    )
    reasons.append(
        {
            "type": "domain",
            "label": "정책 분야 일치",
            "details": evidence["domain_tags"],
        }
    )
    reasons.append(
        {
            "type": "issue",
            "label": "민원 의도 일치",
            "details": evidence["issue_tags"],
        }
    )
    if evidence["qualification_tags"]:
        reasons.append(
            {
                "type": "qualification",
                "label": "배타 자격 조건 일치",
                "details": evidence["qualification_tags"],
            }
        )
    if evidence["age"]["active"]:
        reasons.append(
            {
                "type": "age",
                "label": "정책 연령 규칙과 페르소나 관계 일치",
                "details": {
                    "policy_range": evidence["age"].get("policy_range"),
                    "persona_relationship": evidence["age"].get("persona_relationship"),
                },
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
        source = self.manifest.get("source")
        if isinstance(source, dict) and source:
            return dict(source)
        return load_civil_complaint_source_metadata(self.data_dir)

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
        policy = item.get("policy")
        persona = item.get("persona")
        errors = []
        if not complaint_text:
            errors.append("complaint_text_required")
        if not isinstance(policy, Mapping):
            errors.append("policy_must_be_object")
        if not isinstance(persona, Mapping):
            errors.append("persona_must_be_object")
        if errors:
            return None, errors
        assert isinstance(policy, Mapping)
        assert isinstance(persona, Mapping)

        policy_document = build_policy_context_document(policy)
        if not policy_document:
            errors.append("policy_context_required")
        region, region_error = _policy_region(policy)
        if region is None:
            errors.append(region_error)

        query_document = f"민원: {complaint_text}\n{policy_document}"
        domains = domain_tags(query_document)
        issues = issue_tags(query_document)
        if not domains:
            errors.append("query_domain_unknown")
        if not issues:
            errors.append("query_issue_unknown")

        has_structured_age_contract = all(
            key in policy for key in ("age_min", "age_max")
        )
        structured_age_unrestricted = has_structured_age_contract and all(
            policy.get(key) is None for key in ("age_min", "age_max")
        )
        age_active = not structured_age_unrestricted and (
            bool(POLICY_AGE_CONTEXT_PATTERN.search(query_document))
            or any(policy.get(key) is not None for key in ("age_min", "age_max"))
        )
        age_range = _policy_age_range(policy, policy_document)
        if age_active:
            if age_range is None:
                errors.append("policy_age_range_unknown")
            elif _optional_age(persona.get("age")) is None:
                errors.append("persona_age_unknown")
            elif (
                _persona_age_relationship(
                    _optional_age(persona.get("age")) or 0, age_range
                )
                == "outside"
            ):
                errors.append("persona_outside_policy_or_boundary")

        if errors:
            return None, errors
        assert region is not None
        qualification_document = (
            f"{complaint_text} {policy_document} {_persona_document(persona)}"
        )
        return {
            "complaint_text": complaint_text,
            "policy_document": policy_document,
            "combined_text": query_document,
            "policy": policy,
            "persona": persona,
            "region": region,
            "domains": domains,
            "issues": issues,
            "qualifications": qualification_tags(qualification_document),
            "age_active": age_active,
            "age_range": age_range,
        }, []

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
            policy_documents = [
                prepared[position]["policy_document"]  # type: ignore[index]
                for position in valid_positions
            ]
            embeddings = _embedding_matrix(
                self.embedder,
                complaint_documents + policy_documents,
            )
            collection_count = int(collection.count())
            candidate_count = min(
                max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_COUNT),
                MAX_CANDIDATE_COUNT,
                collection_count,
            )
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

            valid_count = len(valid_positions)
            for batch_index, position in enumerate(valid_positions):
                query = prepared[position]
                assert query is not None
                complaint_scores = _distance_scores(raw, batch_index)
                policy_scores = _distance_scores(raw, valid_count + batch_index)
                candidate_ids = set(complaint_scores) | set(policy_scores)
                rejection_counts = _empty_rejection_counts()
                ranked: list[tuple[float, float, str, dict[str, Any]]] = []

                for case_id in candidate_ids:
                    record = corpus_by_id.get(case_id)
                    if record is None:
                        raise CivilComplaintIndexUnavailableError(
                            f"민원 FAQ Chroma ID가 canonical 정본에 없습니다: {case_id}"
                        )

                    failures, evidence, context = _hard_gate_evidence(query, record)
                    if failures:
                        for failure in failures:
                            rejection_counts[failure] += 1
                        continue

                    complaint_dense = complaint_scores.get(case_id)
                    policy_dense = policy_scores.get(case_id)
                    if complaint_dense is None or policy_dense is None:
                        rejection_counts["missing_dense_score"] += 1
                        continue
                    semantic = 0.55 * complaint_dense + 0.45 * policy_dense
                    candidate_heading = f"{record['title']} {record['question']}"
                    lexical = lexical_score(query["combined_text"], candidate_heading)
                    final = 0.70 * semantic + 0.20 * lexical + 0.10 * context

                    threshold_failed = False
                    for key, score, floor in (
                        (
                            "below_complaint_dense",
                            complaint_dense,
                            COMPLAINT_DENSE_FLOOR,
                        ),
                        ("below_policy_dense", policy_dense, POLICY_DENSE_FLOOR),
                        ("below_semantic", semantic, SEMANTIC_FLOOR),
                    ):
                        if score + SCORE_EPSILON < floor:
                            rejection_counts[key] += 1
                            threshold_failed = True
                    if threshold_failed:
                        continue
                    if final + SCORE_EPSILON < FINAL_SCORE_FLOOR:
                        rejection_counts["below_final"] += 1
                        continue
                    if final + SCORE_EPSILON < UI_REFERENCE_SCORE_FLOOR:
                        rejection_counts["below_ui_threshold"] += 1
                        continue

                    score_payload = _score_payload(
                        complaint_dense=complaint_dense,
                        policy_dense=policy_dense,
                        semantic=semantic,
                        lexical=lexical,
                        context=context,
                        final=final,
                    )
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
                        "confidence": "medium" if final >= 0.78 else "low",
                        "match_reasons": _match_reasons(evidence),
                        "evidence": evidence,
                        "warnings": list(COMMON_WARNINGS),
                    }
                    ranked.append((final, semantic, case_id, result))

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
    "POLICY_DENSE_FLOOR",
    "SEMANTIC_FLOOR",
    "FINAL_SCORE_FLOOR",
    "UI_REFERENCE_SCORE_FLOOR",
    "ACTIVE_POINTER_FILENAME",
    "ACTIVE_POINTER_SCHEMA_VERSION",
    "ACTIVE_RELOAD_STRATEGY",
    "find_similar_complaint_cases_batch",
    "get_civil_complaint_similarity_service",
    "resolve_active_index_dir",
]
