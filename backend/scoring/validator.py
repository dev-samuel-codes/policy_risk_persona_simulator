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

    # 4) 배제 사유 오류 체크 (정책이 나이 조건일 때만)
    try:
        age_int = int(persona.get("age", -1))
    except (ValueError, TypeError):
        age_int = -1

    is_out_of_age_range = age_int != -1 and not (19 <= age_int <= 34)

    if is_out_of_age_range:
        age_keywords = ["나이", "세", "청년", "연령", "34"]
        mentions_age_reason = any(
            any(kw in c.get("complaint_text", "") for kw in age_keywords)
            for c in complaints
        )
        if not mentions_age_reason:
            errors.append("나이 대상 밖인데 배제 사유에 나이 언급이 전혀 없음 (엉뚱한 사유일 가능성)")

    return errors
