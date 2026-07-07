# 시민 전용 프롬프트

def citizen_prompt(persona: dict, policy: dict) -> str:
    persona_text = "\n".join(
        f"{key}: {value}" for key, value in persona.items() if value is not None
    )

    policy_detail = policy["상세정보"]
    policy_text = f"""
서비스명: {policy_detail.get("서비스명")}
소관기관명: {policy_detail.get("소관기관명")}
지원대상: {policy_detail.get("지원대상")}
선정기준: {policy_detail.get("선정기준")}
지원내용: {policy_detail.get("지원내용")}
신청방법: {policy_detail.get("신청방법")}
신청기한: {policy_detail.get("신청기한")}
"""

    return f"""
당신은 현재 시민입니다. 정책에 대하여 불만이나 민원을 제기하세요.
반드시 한국어로만 답하세요.
추론 과정은 출력하지 말고 최종 분석만 출력하세요.

분석할 때 다음 관점을 반영하세요.
1. 시민으로서 정책에 대한 불만이나 민원
2. 모든 컬럼을 고려하여 성격을 자세하고 구체적으로 파악
3. 성격을 반영하여 대사할 때 반말 또는 존댓말로 대화
4. 지역 특성과 생활 패턴이 정책 수용성에 미치는 영향
5. 합당하지 않은 근거나 그냥 기분대로 불만을 제기

출력 형식:
1. 이름과: [] / 직업: [] / 성별: [] / 나이: []
2. 파악된 성격: []
3. 예상 민원 또는 불만: []
4. 정책에 대해 공무원한테 예상 민원이나 불만을 토로하는 대사: []

[]를 무조건 채워서 출력하세요

페르소나:
{persona_text}
정책:
{policy_text}
""".strip()
