import re


def validate_citizen_response(result: dict, persona: dict, policy: dict) -> list[str]:
    """
    검증 실패 사유 목록을 반환. 빈 리스트면 통과.
    """
    errors = []

    if result is None:
        return ["파싱 실패 (None)"]

    # 1) 나이 불일치 체크
    summary_age_raw = result.get("persona_summary", {}).get("나이", "")
    summary_age = re.sub(r"[^0-9]", "", str(summary_age_raw))
    actual_age = str(persona.get("age", ""))
    if summary_age and actual_age and summary_age != actual_age:
        errors.append(f"나이 불일치: persona_summary={summary_age}, 실제={actual_age}")

    # 2) 이름 누락 체크
    name = result.get("persona_summary", {}).get("이름", "")
    if not name.strip():
        errors.append("이름 필드가 비어있음")

    # 3) JSON 영문 키를 제외하고 모델이 생성한 실제 문장만 영어 혼용 체크
    persona_summary = result.get("persona_summary", {})
    complaints = result.get("complaints", [])
    generated_texts = [
        *(str(value) for value in persona_summary.values()),
        str(result.get("personality", "")),
        *(str(complaint.get("complaint_text", "")) for complaint in complaints),
        *(str(complaint.get("dialogue", "")) for complaint in complaints),
    ]
    full_text = " ".join(generated_texts)
    if re.search(r"[a-zA-Z]{3,}", full_text):
        errors.append("영어 단어 혼용 감지")

    return errors
