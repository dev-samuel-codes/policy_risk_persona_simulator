from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import (
    run_llm,
)
from backend.scoring.risk_keywords import RISK_KEYWORDS


def classify_by_rule(complaint_text: str) -> str | None:
    scores = {}
    for category_id, keywords in RISK_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in complaint_text)
        if match_count > 0:
            scores[category_id] = match_count
    if not scores:
        return None
    return max(scores, key=scores.get)


def classify_by_llm(complaint_text: str) -> str:
    prompt = f"""
다음 민원 문장을 아래 8개 카테고리 중 가장 가까운 것 하나로만 분류하세요.
반드시 카테고리 id만 출력하세요 (설명 없이 id 한 단어만).
카테고리 id는 분류 코드이므로 영어 표기를 그대로 사용하세요.

카테고리:
- target_ambiguous : 대상 조건 모호
- access_barrier   : 신청 접근성 문제
- document_burden  : 서류 부담
- info_gap         : 정보 격차
- equity_issue     : 형평성 논란
- regional_gap     : 지역 격차
- digital_divide   : 디지털 취약
- complaint_surge  : 민원 폭증 가능

민원 문장: "{complaint_text}"

출력 (id만):
""".strip()

    # 시민 응답 생성과 같은 로컬 Qwen 모델을 사용하여 민원 분류
    result = run_llm(prompt).strip()
    valid_ids = set(RISK_KEYWORDS.keys())
    return result if result in valid_ids else "complaint_surge"


def classify_complaint(complaint_text: str) -> str:
    rule_result = classify_by_rule(complaint_text)
    if rule_result:
        return rule_result
    return classify_by_llm(complaint_text)


def add_risk_categories(simulation_results: list[dict]) -> list[dict]:
    for persona_result in simulation_results:
        for complaint in persona_result["complaints"]:
            complaint["risk_category"] = classify_complaint(complaint["complaint_text"])
    return simulation_results
