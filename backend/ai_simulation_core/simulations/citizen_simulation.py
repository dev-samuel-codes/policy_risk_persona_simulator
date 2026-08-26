import json
import re
import unicodedata

from backend.ai_simulation_core.llm.llm_gateway import (
    run_llm,
)
from backend.ai_simulation_core.prompts.citizen_prompt import citizen_prompt
from backend.ai_simulation_core.simulations.citizen_quality import (
    ALLOWED_COMPLAINT_BASES,
    normalize_maximum_support_qualifiers,
    validate_semantic_quality,
)


_COMPLAINT_BASIS_ALIASES = {
    "신청절차": "신청방법",
    "접수방법": "신청방법",
    "신청기간": "신청기한",
    "접수기간": "신청기한",
    "제출서류": "구비서류",
    "필요서류": "구비서류",
    "지원혜택": "지원내용",
    "지원금": "지원내용",
    "연락처": "문의처",
    "문의기관": "문의처",
    "소관기관명": "정보미제공",
    "정보부족": "정보미제공",
    "정보누락": "정보미제공",
    "개인사정": "개인상황",
    "개인여건": "개인상황",
}


def _canonicalize_complaint_basis(value: object) -> object:
    if not isinstance(value, str):
        return value
    compact = re.sub(
        r"[\s\-_·]",
        "",
        unicodedata.normalize("NFKC", value),
    )
    if compact in ALLOWED_COMPLAINT_BASES:
        return compact
    return _COMPLAINT_BASIS_ALIASES.get(compact, value.strip())


def _normalize_complaint_bases(result: dict) -> dict:
    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return result
    for complaint in complaints:
        if isinstance(complaint, dict) and "basis" in complaint:
            complaint["basis"] = _canonicalize_complaint_basis(complaint["basis"])
    return result


def parse_citizen_response(raw_output: str) -> dict | None:
    # 로컬 모델이 JSON 코드 블록을 붙인 경우 마크다운 기호를 제거
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"[JSON 파싱 실패] {error}\n원본: {raw_output[:200]}")
        return None

    if not isinstance(parsed, dict):
        print("[JSON 구조 오류] 최상위 값은 객체여야 합니다.")
        return None
    return _normalize_complaint_bases(parsed)


def validate_citizen_response(
    result: dict,
    persona: dict,
    policy: dict | None = None,
) -> list[str]:
    """
    검증 실패 사유 목록을 반환. 빈 리스트면 통과.
    """
    errors = []

    if result is None:
        return ["파싱 실패 (None)"]

    persona_summary = result.get("persona_summary", {})
    if not isinstance(persona_summary, dict):
        errors.append("persona_summary가 JSON 객체가 아님")
        persona_summary = {}

    # 1) 나이 필수값과 불일치 체크
    summary_age_raw = persona_summary.get("나이", "")
    summary_age_match = re.fullmatch(
        r"\s*(\d{1,3})(?:\s*세)?\s*",
        str(summary_age_raw),
    )
    summary_age = summary_age_match.group(1) if summary_age_match else ""
    actual_age_raw = persona.get("age")
    actual_age = "" if actual_age_raw is None else str(actual_age_raw)
    if actual_age and not summary_age:
        errors.append("나이 필드가 비어있음")
    elif summary_age and actual_age and summary_age != actual_age:
        errors.append(f"나이 불일치: persona_summary={summary_age}, 실제={actual_age}")

    # 2) 이름 누락 체크
    name = persona_summary.get("이름", "")
    if not isinstance(name, str) or not name.strip():
        errors.append("이름 필드가 비어있거나 문자열이 아님")

    # 3) 성격 및 민원 필수 구조 체크
    personality = result.get("personality", "")
    if not isinstance(personality, str) or not personality.strip():
        errors.append("personality가 비어있거나 문자열이 아님")

    complaints = result.get("complaints", [])
    if not isinstance(complaints, list):
        errors.append("complaints가 JSON 배열이 아님")
        complaints = []
    elif len(complaints) != 1:
        errors.append("complaints는 정확히 1개여야 함")

    valid_complaints = []
    for index, complaint in enumerate(complaints, start=1):
        if not isinstance(complaint, dict):
            errors.append("complaints 항목이 JSON 객체가 아님")
            continue
        valid_complaints.append(complaint)
        for field in ("complaint_text", "dialogue"):
            value = complaint.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"complaints[{index}].{field}가 비어있거나 문자열이 아님")

    # 4) JSON 영문 키를 제외하고 모델이 생성한 실제 문장만 영어 혼용 체크
    generated_texts = [
        str(name),
        str(personality),
        *(str(complaint.get("complaint_text", "")) for complaint in valid_complaints),
        *(str(complaint.get("dialogue", "")) for complaint in valid_complaints),
    ]
    full_text = " ".join(generated_texts)
    if re.search(r"[a-zA-Z]", full_text):
        errors.append("영어 단어 혼용 감지")

    if policy is not None:
        errors.extend(validate_semantic_quality(result, persona, policy))

    return list(dict.fromkeys(errors))


def run_citizen_simulation(
    persona: dict, policy: dict, max_retries: int = 3
) -> dict | None:
    parsed = None
    errors = []
    validation_feedback = None

    for attempt in range(max_retries):
        prompt = citizen_prompt(
            persona,
            policy,
            validation_feedback=validation_feedback,
        )
        # main과 같은 로컬 Qwen 모델을 사용하여 시민 응답 생성
        raw_output = run_llm(prompt)
        parsed = parse_citizen_response(raw_output)

        if parsed is None:
            print(f"  [시도 {attempt + 1}/{max_retries}] JSON 파싱 실패, 재시도")
            errors = ["JSON_PARSE_ERROR"]
            validation_feedback = errors
            continue

        parsed = normalize_maximum_support_qualifiers(parsed, policy)
        errors = validate_citizen_response(parsed, persona, policy)
        if not errors:
            if attempt > 0:
                print(f"  [시도 {attempt + 1}/{max_retries}] 검증 통과")
            break
        print(f"  [시도 {attempt + 1}/{max_retries}] 검증 실패: {errors}")
        validation_feedback = errors

    if parsed is None or errors:
        if errors:
            print(f"  [최종 검증 실패] 유효한 시민 응답을 생성하지 못했습니다: {errors}")
        return None

    # 검증을 통과한 결과에만 스코어링용 페르소나 식별자를 추가한다.
    parsed["persona_id"] = persona.get("uuid")
    parsed["_validation_errors"] = []
    parsed["_quality_gate"] = {
        "version": "citizen-grounding-v1",
        "status": "passed",
        "removed_complaints": 0,
        "generation_attempts": attempt + 1,
    }
    return parsed
