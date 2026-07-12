import json
import re

from backend.ai_simulation_core.llm.llm_gateway import (
    run_llm,
)
from backend.ai_simulation_core.prompts.citizen_prompt import citizen_prompt


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
    return parsed


def validate_citizen_response(result: dict, persona: dict) -> list[str]:
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

    # 1) 나이 불일치 체크
    summary_age_raw = persona_summary.get("나이", "")
    summary_age = re.sub(r"[^0-9]", "", str(summary_age_raw))
    actual_age = str(persona.get("age", ""))
    if summary_age and actual_age and summary_age != actual_age:
        errors.append(f"나이 불일치: persona_summary={summary_age}, 실제={actual_age}")

    # 2) 이름 누락 체크
    name = persona_summary.get("이름", "")
    if not isinstance(name, str) or not name.strip():
        errors.append("이름 필드가 비어있거나 문자열이 아님")

    # 3) JSON 영문 키를 제외하고 모델이 생성한 실제 문장만 영어 혼용 체크
    complaints = result.get("complaints", [])
    if not isinstance(complaints, list):
        errors.append("complaints가 JSON 배열이 아님")
        complaints = []

    valid_complaints = []
    for complaint in complaints:
        if not isinstance(complaint, dict):
            errors.append("complaints 항목이 JSON 객체가 아님")
            continue
        valid_complaints.append(complaint)

    generated_texts = [
        *(str(value) for value in persona_summary.values()),
        str(result.get("personality", "")),
        *(str(complaint.get("complaint_text", "")) for complaint in valid_complaints),
        *(str(complaint.get("dialogue", "")) for complaint in valid_complaints),
    ]
    full_text = " ".join(generated_texts)
    if re.search(r"[a-zA-Z]{3,}", full_text):
        errors.append("영어 단어 혼용 감지")

    return errors


def run_citizen_simulation(
    persona: dict, policy: dict, max_retries: int = 3
) -> dict | None:
    prompt = citizen_prompt(persona, policy)

    parsed = None
    errors = []

    for attempt in range(max_retries):
        # main과 같은 로컬 Qwen 모델을 사용하여 시민 응답 생성
        raw_output = run_llm(prompt)
        parsed = parse_citizen_response(raw_output)

        if parsed is None:
            print(f"  [시도 {attempt + 1}/{max_retries}] JSON 파싱 실패, 재시도")
            continue

        errors = validate_citizen_response(parsed, persona)
        if not errors:
            if attempt > 0:
                print(f"  [시도 {attempt + 1}/{max_retries}] 검증 통과")
            break
        print(f"  [시도 {attempt + 1}/{max_retries}] 검증 실패: {errors}")

    if parsed is None:
        return None

    # 스코어링과 결과 확인에 필요한 페르소나 식별자와 검증 결과 추가
    parsed["persona_id"] = persona.get("id")
    parsed["_validation_errors"] = errors
    return parsed
