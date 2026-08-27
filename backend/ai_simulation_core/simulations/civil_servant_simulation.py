"""Qwen 공무원 답변을 생성하고 시민·공무원 연결 구조를 검증한다."""

from __future__ import annotations

import json
import re
from typing import Any

from backend.ai_simulation_core.llm.llm_gateway import run_llm
from backend.ai_simulation_core.prompts.civil_servant_prompt import (
    civil_servant_prompt,
)
from backend.ai_simulation_core.simulations.citizen_quality import (
    validate_policy_grounded_text,
)


QUALITY_MODE = "qwen_best_effort_v1"
INACTIVE_OFFICIAL_MARKERS = ("전직", "퇴직", "은퇴", "구직")
_OFFICIAL_COMMITMENT_PATTERN = re.compile(
    r"(?:승인|선정|지급|수급|연장)"
    r"(?:하겠습니다|해\s*드리겠습니다|되도록\s*하겠습니다|"
    r"을\s*(?:보장|확정|처리)하겠습니다)|"
    r"(?:지원\s*)?대상자(?:로)?\s*(?:처리|등록|확정)"
    r"(?:하겠습니다|해\s*드리겠습니다)"
)
_OFFICIAL_RISK_SCORING_PATTERN = re.compile(
    r"민원\s*(?:점수|위험도|리스크)|"
    r"(?:리스크|위험도)(?:\s*(?:점수|평가|등급))?|"
    r"정책\s*집행\s*(?:평가|위험|리스크)|"
    r"(?:위험|리스크)\s*점수|점수화"
)
_BASIS_RESPONSE_ANCHORS = {
    "지원대상": re.compile(r"지원\s*대상|대상자|자격"),
    "선정기준": re.compile(r"선정\s*기준|심사\s*기준"),
    "지원내용": re.compile(
        r"지원\s*(?:내용|규모|금액|혜택|기간|여부)|지원금|실효성|주거\s*부담"
    ),
    "신청방법": re.compile(
        r"신청\s*(?:방법|절차|경로|방식)|접수\s*(?:방법|경로)|"
        r"온라인\s*신청|방문\s*신청"
    ),
    "신청기한": re.compile(
        r"신청\s*(?:기한|기간)|접수\s*(?:기한|기간)|마감|"
        r"기한\s*변경|준비\s*기간"
    ),
    "구비서류": re.compile(
        r"구비\s*서류|제출\s*서류|필요(?:한)?\s*서류|신청서류|"
        r"서류\s*(?:발급|제출|준비)"
    ),
    "제외조건": re.compile(r"제외\s*(?:조건|대상)|탈락\s*조건"),
    "문의처": re.compile(
        r"문의처|연락처|문의\s*(?:기관|방법)|담당\s*(?:기관|부서)"
    ),
    "정보미제공": re.compile(
        r"명시되어\s*있지|정보(?:가|는)?\s*없|정보\s*미제공|"
        r"입력에\s*없|제공된\s*내용만으로"
    ),
    "개인상황": re.compile(
        r"개인\s*(?:상황|사정)|전체\s*자격|실제\s*(?:선정|승인)|공식\s*심사"
    ),
}
_SAFE_GENERAL_RESPONSE_PATTERN = re.compile(
    r"민원(?:\s*내용)?(?:을|이)?.{0,12}(?:확인|검토)|"
    r"(?:입력된|제공된)\s*(?:정책|내용).{0,36}(?:확정할\s*수\s*없|확인\s*필요)|"
    r"정책\s*원문.{0,24}(?:확인|근거)"
)
_STRUCTURED_FRAGMENT_PATTERN = re.compile(
    r"```|</?[^<>\n]+>|[{}]|"
    r"[\"']?response[\"']?\s*:|"
    r"\[\s*(?:\{|\[|\"|'|-?\d|true\b|false\b|null\b)",
    re.IGNORECASE,
)

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


def parse_civil_servant_response(raw_output: str) -> str | None:
    """Qwen의 JSON·코드 펜스·평문 출력을 답변 본문 문자열로 정규화한다."""

    if not isinstance(raw_output, str):
        return None
    cleaned = raw_output.strip()
    if not cleaned:
        return None
    was_fenced = False
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        was_fenced = True
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        if was_fenced or _STRUCTURED_FRAGMENT_PATTERN.search(cleaned):
            return None
        return cleaned

    if isinstance(parsed, dict):
        response = parsed.get("response")
        return (
            response.strip()
            if isinstance(response, str) and response.strip()
            else None
        )
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return None


def _citizen_complaint_text(citizen_result: dict) -> str:
    complaint = _complaint(citizen_result)
    return "\n".join(
        f"{label}: {_text(complaint.get(field))}"
        for field, label in (
            ("complaint_text", "민원 요지"),
            ("dialogue", "민원 발언"),
        )
        if _text(complaint.get(field))
    )


def _response_addresses_basis(response: str, basis: str) -> bool:
    expected_pattern = _BASIS_RESPONSE_ANCHORS.get(basis)
    if expected_pattern is None:
        return False
    if expected_pattern.search(response):
        return True

    # 다른 구체 필드만 답한 경우 일반적인 확인 문구가 있어도 통과시키지 않는다.
    concrete_bases = (
        "지원대상",
        "선정기준",
        "지원내용",
        "신청방법",
        "신청기한",
        "구비서류",
        "제외조건",
        "문의처",
    )
    if any(
        other_basis != basis and _BASIS_RESPONSE_ANCHORS[other_basis].search(response)
        for other_basis in concrete_bases
    ):
        return False
    return bool(_SAFE_GENERAL_RESPONSE_PATTERN.search(response))


def validate_civil_servant_response(
    result: object,
    *,
    persona: dict,
    policy: dict,
    citizen_result: dict,
    enforce_content_quality: bool = True,
) -> list[str]:
    if not isinstance(result, dict):
        return ["OFFICIAL_RESULT_NOT_OBJECT"]

    errors: list[str] = []
    official_id = _text(persona.get("uuid"))
    occupation = _text(persona.get("occupation"))
    citizen_id = _citizen_persona_id(citizen_result)
    expected_basis, _ = build_grounded_response(policy, citizen_result)

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
    response = _text(result.get("response"))
    if not response:
        errors.append("OFFICIAL_RESPONSE_EMPTY")
    elif enforce_content_quality:
        citizen_persona = citizen_result.get("persona")
        if not isinstance(citizen_persona, dict):
            citizen_persona = {}
        errors.extend(
            validate_policy_grounded_text(
                response,
                citizen_persona,
                policy,
                path="official.response",
            )
        )
        if not _response_addresses_basis(response, expected_basis):
            errors.append("OFFICIAL_BASIS_UNADDRESSED")
        if _OFFICIAL_COMMITMENT_PATTERN.search(response):
            errors.append("UNSUPPORTED_OFFICIAL_COMMITMENT")
        if _OFFICIAL_RISK_SCORING_PATTERN.search(response):
            errors.append("OFFICIAL_RISK_SCORING_CONTENT")
    if result.get("_validation_errors") != []:
        errors.append("OFFICIAL_VALIDATION_ERRORS_NOT_EMPTY")
    gate = result.get("_quality_gate")
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        errors.append("OFFICIAL_QUALITY_GATE_NOT_PASSED")
    elif (
        gate.get("mode") != QUALITY_MODE
        or gate.get("removed_statements") != 0
        or not isinstance(gate.get("generation_attempts"), int)
        or not 1 <= gate["generation_attempts"] <= 3
    ):
        errors.append("OFFICIAL_QUALITY_GATE_INVALID")
    return list(dict.fromkeys(errors))


def run_civil_servant_simulation(
    persona: dict,
    policy: dict,
    citizen_result: dict,
    max_retries: int = 3,
) -> dict:
    """Qwen을 한 번 호출하고, 사용할 답변이 없으면 서버 기본 답변을 반환한다."""

    if not 1 <= max_retries <= 3:
        raise ValueError("공무원 응답 재시도 횟수는 1 이상 3 이하여야 합니다.")
    basis, grounded_response = build_grounded_response(policy, citizen_result)
    official_id = _text(persona.get("uuid"))
    citizen_id = _citizen_persona_id(citizen_result)
    occupation = _text(persona.get("occupation"))
    input_errors = []
    if not official_id:
        input_errors.append("OFFICIAL_PERSONA_ID_MISMATCH")
    if "공무원" not in occupation or any(
        marker in occupation for marker in INACTIVE_OFFICIAL_MARKERS
    ):
        input_errors.append("OFFICIAL_PERSONA_NOT_ACTIVE")
    if not citizen_id:
        input_errors.append("OFFICIAL_CITIZEN_LINK_MISMATCH")
    if input_errors:
        raise RuntimeError("공무원 응답 입력 검증 실패: " + ", ".join(input_errors))

    complaint_text = _citizen_complaint_text(citizen_result)
    prompt = civil_servant_prompt(
        persona,
        policy,
        complaint_text,
        grounded_response,
        validation_feedback=None,
    )
    fallback_used = False
    try:
        raw_output = run_llm(prompt)
        response = parse_civil_servant_response(raw_output)
    except Exception as exc:
        response = None
        print(
            "  [공무원 생성] Qwen 호출 실패, 서버 기본 답변 사용: "
            f"{type(exc).__name__}"
        )

    if response is None:
        response = grounded_response
        fallback_used = True
        print("  [공무원 생성] 사용할 응답이 없어 서버 기본 답변을 사용합니다.")

    result = {
        "official_persona_id": official_id,
        "citizen_persona_id": citizen_id,
        "basis": basis,
        "response": response,
        "_validation_errors": [],
        "_quality_gate": {
            "status": "passed",
            "mode": QUALITY_MODE,
            "removed_statements": 0,
            "generation_attempts": 1,
            "fallback_used": fallback_used,
        },
    }
    structural_errors = validate_civil_servant_response(
        result,
        persona=persona,
        policy=policy,
        citizen_result=citizen_result,
        enforce_content_quality=False,
    )
    if structural_errors:
        raise RuntimeError(
            "공무원 응답 구조 검증 실패: " + ", ".join(structural_errors)
        )
    return result
