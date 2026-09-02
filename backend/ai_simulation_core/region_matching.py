"""정책과 페르소나의 구조화 지역 조건을 동일한 규칙으로 비교한다."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalize_region(value: Any) -> str:
    return re.sub(
        r"[\s\-_·]",
        "",
        unicodedata.normalize("NFKC", str(value or "")),
    )


def _district_without_province(district: Any, province: Any) -> str:
    district_text = unicodedata.normalize("NFKC", str(district or "")).strip()
    normalized_province = _normalize_region(province)
    prefixed = re.fullmatch(r"\s*([^\s\-_·]+)\s*[\-_·]\s*(.+?)\s*", district_text)
    if prefixed and _normalize_region(prefixed.group(1)) == normalized_province:
        return _normalize_region(prefixed.group(2))
    return _normalize_region(district_text)


def region_matches(
    persona: dict,
    *,
    region_scope: str,
    province: str = "",
    district: str = "",
) -> bool:
    if region_scope == "nationwide":
        return True
    if region_scope != "specific":
        return False

    policy_province = _normalize_region(province)
    persona_province = _normalize_region(persona.get("province"))
    if not policy_province or policy_province != persona_province:
        return False

    policy_district = _district_without_province(district, policy_province)
    if not policy_district:
        return True
    persona_district = _district_without_province(
        persona.get("district"),
        persona_province,
    )
    return policy_district == persona_district
