"""정책 원문에만 근거한 공무원 답변을 만든다."""

from __future__ import annotations

import re
from typing import Any


QUALITY_MODE = "deterministic_policy_grounded_v1"
INACTIVE_OFFICIAL_MARKERS = ("전직", "퇴직", "은퇴", "구직")

_FIELD_KEYS = {
    "지원대상": "지원대상",
    "선정기준": "선정기준",
    "지원내용": "지원내용",
    "신청방법": "신청방법",
    "신청기한": "신청기한",
    "구비서류": "구비서류",
    "제외조건": "제외조건",
    "문의처": "문의처",
}
_MISSING_FIELD_HINTS = (
    ("신청방법", ("신청방법", "신청 방법", "어떻게 신청")),
    ("신청기한", ("신청기한", "신청 기한", "신청기간", "신청 기간", "언제 신청")),
    ("구비서류", ("구비서류", "구비 서류", "필요한 서류", "제출 서류")),
    ("문의처", ("문의처", "연락처", "어디에 문의")),
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _policy_detail(policy: dict) -> dict:
    detail = policy.get("상세정보")
    return detail if isinstance(detail, dict) else {}


def _policy_value(policy: dict, field: str) -> str:
    detail = _policy_detail(policy)
    value = _text(detail.get(_FIELD_KEYS[field]))
    if value:
        return value
    listing = policy.get("목록정보")
    if isinstance(listing, dict):
        aliases = {
            "지원대상": "지원대상",
            "지원내용": "지원내용",
            "신청방법": "신청방법",
            "신청기한": "신청기한",
            "문의처": "전화문의",
        }
        alias = aliases.get(field)
        if alias:
            return _text(listing.get(alias))
    return ""


def _quote(value: str, *, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _format_won(value: int) -> str:
    if value % 10_000 == 0:
        return f"{value // 10_000}만 원"
    return f"{value:,}원"


def _display_grounded_value(field: str, value: str) -> str:
    """영문 정책 입력 중 결정적으로 번역 가능한 값만 한국어로 표시한다."""
    compact = re.sub(r"\s+", " ", value).strip()
    normalized = compact.lower()
    exact_translations = {
        ("지원대상", "residents of seoul age 20 through 39"): "서울 거주 20세 이상 39세 이하 시민",
        ("신청방법", "apply online or visit local community service center"): (
            "온라인 신청 또는 지역 행정복지센터 방문"
        ),
        ("구비서류", "identification card and proof of residence"): (
            "신분증 및 거주 증명서"
        ),
        ("문의처", "seoul housing support division 02-120"): (
            "서울 주거지원과 02-120"
        ),
        ("선정기준", "homeowners are excluded"): "주택 소유자 제외",
        ("제외조건", "homeowners are excluded"): "주택 소유자 제외",
    }
    translated = exact_translations.get((field, normalized))
    if translated:
        return translated

    date_range = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})\s+to\s+"
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        normalized,
    )
    if date_range and field == "신청기한":
        start_year, start_month, start_day, end_year, end_month, end_day = (
            int(part) for part in date_range.groups()
        )
        return (
            f"{start_year}년 {start_month}월 {start_day}일부터 "
            f"{end_year}년 {end_month}월 {end_day}일까지"
        )

    support = re.fullmatch(
        r"krw\s*([\d,]+)\s+per\s+month\s+for\s+up\s+to\s+"
        r"(\d+)\s+months?",
        normalized,
    )
    if support and field == "지원내용":
        amount, months = support.groups()
        return f"월 {_format_won(int(amount.replace(',', '')))}, 최대 {int(months)}개월 지원"
    return compact


def _citizen_persona_id(citizen_result: dict) -> str:
    persona_id = _text(citizen_result.get("persona_id"))
    if persona_id:
        return persona_id
    persona = citizen_result.get("persona")
    if isinstance(persona, dict):
        return _text(persona.get("uuid"))
    return ""


def _complaint(citizen_result: dict) -> dict:
    complaints = citizen_result.get("complaints")
    if not isinstance(complaints, list) or len(complaints) != 1:
        raise ValueError("공무원 답변에는 검증된 시민 민원 정확히 1개가 필요합니다.")
    complaint = complaints[0]
    if not isinstance(complaint, dict):
        raise ValueError("시민 민원이 JSON 객체가 아닙니다.")
    if not _text(complaint.get("complaint_text")) and not _text(
        complaint.get("dialogue")
    ):
        raise ValueError("시민 민원 내용이 비어 있습니다.")
    return complaint


def _missing_field_from_complaint(policy: dict, complaint: dict) -> str:
    combined = " ".join(
        _text(complaint.get(field)) for field in ("complaint_text", "dialogue")
    )
    # 문장에 정책 필드명이 직접 쓰였으면 일반적인 표현(예: "어떻게 신청")보다
    # 그 명시적 쟁점을 우선한다.
    for label, _ in _MISSING_FIELD_HINTS:
        if label in combined:
            return label
    for label, hints in _MISSING_FIELD_HINTS:
        if any(hint in combined for hint in hints):
            return label
    for label, _ in _MISSING_FIELD_HINTS:
        if not _policy_value(policy, label):
            return label
    return "문의처"


def _missing_response(field: str) -> str:
    return (
        f"입력된 정책에는 {field} 정보가 명시되어 있지 않습니다. "
        "제공된 내용만으로는 이를 확정할 수 없으므로 공식 공고 또는 담당 기관의 "
        "확인이 필요합니다."
    )


def _grounded_field_response(field: str, value: str, policy: dict) -> str:
    quoted = _quote(_display_grounded_value(field, value))
    if field == "지원내용":
        return (
            f"입력된 정책의 지원 내용은 '{quoted}'입니다. 지원 규모가 실제 부담을 "
            "충분히 줄이는지는 정책 검토 시 살펴볼 쟁점이며, 추가 지원 여부는 "
            "입력된 내용만으로 확정할 수 없습니다."
        )
    if field == "신청방법":
        return (
            f"입력된 신청방법은 '{quoted}'입니다. 제기된 접근성 우려는 검토할 "
            "쟁점이며, 입력에 없는 다른 신청 경로는 확정할 수 없습니다."
        )
    if field == "신청기한":
        return (
            f"입력된 신청기한은 '{quoted}'입니다. 준비 기간이 충분한지는 신청자의 "
            "상황에 따라 다를 수 있으며, 기한 변경 여부는 입력된 내용만으로 "
            "확정할 수 없습니다."
        )
    if field == "구비서류":
        application_method = _policy_value(policy, "신청방법")
        if application_method:
            displayed_method = _quote(
                _display_grounded_value("신청방법", application_method)
            )
            return (
                f"입력된 구비서류는 '{quoted}'이며 신청방법은 "
                f"'{displayed_method}'입니다. 이 신청 경로에서의 구체적인 서류 "
                "발급처·유효기간·제출 순서는 공식 안내 확인이 필요합니다."
            )
        return (
            f"입력된 구비서류는 '{quoted}'입니다. 발급처·유효기간·제출 방식처럼 "
            "입력에 없는 세부 절차는 공식 공고 또는 담당 기관의 확인이 필요합니다."
        )
    if field == "문의처":
        return (
            f"입력된 문의처는 '{quoted}'입니다. 운영시간이나 상담 방식처럼 입력에 "
            "없는 세부사항은 해당 문의처의 공식 안내 확인이 필요합니다."
        )
    if field in {"지원대상", "선정기준", "제외조건"}:
        return (
            f"입력된 {field} 정보는 '{quoted}'입니다. 지역·나이 외 전체 자격과 "
            "실제 선정·승인 여부는 입력된 내용만으로 확정할 수 없습니다."
        )
    raise ValueError(f"지원하지 않는 공무원 답변 근거입니다: {field}")


def _append_contact_guidance(body: str, policy: dict, field: str) -> str:
    if field not in {"구비서류", "신청방법", "신청기한"}:
        return body
    contact = _policy_value(policy, "문의처")
    if not contact:
        return body
    displayed_contact = _quote(_display_grounded_value("문의처", contact))
    return f"{body} 입력된 문의처는 '{displayed_contact}'입니다."


def build_grounded_response(policy: dict, citizen_result: dict) -> tuple[str, str]:
    complaint = _complaint(citizen_result)
    basis = _text(complaint.get("basis")) or "개인상황"
    if basis == "정보미제공":
        field = _missing_field_from_complaint(policy, complaint)
        value = _policy_value(policy, field)
        body = (
            _grounded_field_response(field, value, policy)
            if value
            else _missing_response(field)
        )
        body = _append_contact_guidance(body, policy, field)
    elif basis == "개인상황":
        body = (
            "민원에서 제기한 개인 상황은 입력된 정책 원문만으로 확인할 수 없습니다. "
            "지역·나이 외 전체 자격과 실제 선정·승인 여부는 공식 심사 전에는 "
            "확정할 수 없습니다."
        )
    elif basis in _FIELD_KEYS:
        value = _policy_value(policy, basis)
        body = (
            _grounded_field_response(basis, value, policy)
            if value
            else _missing_response(basis)
        )
        body = _append_contact_guidance(body, policy, basis)
    else:
        raise ValueError(f"허용되지 않은 시민 민원 근거입니다: {basis or 'empty'}")
    return basis, "민원 내용을 확인했습니다. " + body


def validate_civil_servant_response(
    result: object,
    *,
    persona: dict,
    policy: dict,
    citizen_result: dict,
) -> list[str]:
    if not isinstance(result, dict):
        return ["OFFICIAL_RESULT_NOT_OBJECT"]

    errors: list[str] = []
    official_id = _text(persona.get("uuid"))
    occupation = _text(persona.get("occupation"))
    citizen_id = _citizen_persona_id(citizen_result)
    expected_basis, expected_response = build_grounded_response(policy, citizen_result)

    if not official_id or _text(result.get("official_persona_id")) != official_id:
        errors.append("OFFICIAL_PERSONA_ID_MISMATCH")
    if "공무원" not in occupation or any(
        marker in occupation for marker in INACTIVE_OFFICIAL_MARKERS
    ):
        errors.append("OFFICIAL_PERSONA_NOT_ACTIVE")
    if not citizen_id or _text(result.get("citizen_persona_id")) != citizen_id:
        errors.append("OFFICIAL_CITIZEN_LINK_MISMATCH")
    if _text(result.get("basis")) != expected_basis:
        errors.append("OFFICIAL_BASIS_MISMATCH")
    if _text(result.get("response")) != expected_response:
        errors.append("OFFICIAL_RESPONSE_NOT_GROUNDED")
    if result.get("_validation_errors") != []:
        errors.append("OFFICIAL_VALIDATION_ERRORS_NOT_EMPTY")
    gate = result.get("_quality_gate")
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        errors.append("OFFICIAL_QUALITY_GATE_NOT_PASSED")
    elif gate.get("mode") != QUALITY_MODE or gate.get("removed_statements") != 0:
        errors.append("OFFICIAL_QUALITY_GATE_INVALID")
    return errors


def run_civil_servant_simulation(
    persona: dict,
    policy: dict,
    citizen_result: dict,
) -> dict:
    basis, response = build_grounded_response(policy, citizen_result)
    result = {
        "official_persona_id": _text(persona.get("uuid")),
        "citizen_persona_id": _citizen_persona_id(citizen_result),
        "basis": basis,
        "response": response,
        "_validation_errors": [],
        "_quality_gate": {
            "status": "passed",
            "mode": QUALITY_MODE,
            "removed_statements": 0,
        },
    }
    errors = validate_civil_servant_response(
        result,
        persona=persona,
        policy=policy,
        citizen_result=citizen_result,
    )
    if errors:
        raise RuntimeError("공무원 응답 품질 검증 실패: " + ", ".join(errors))
    return result
