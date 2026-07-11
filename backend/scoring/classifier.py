import ollama
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


def classify_by_llm(complaint_text: str, model: str = "qwen2.5:7b") -> str:
    prompt = f"""
다음 민원 문장을 아래 8개 카테고리 중 가장 가까운 것 하나로만 분류하세요.
반드시 카테고리 id만 출력하세요 (설명 없이 id 한 단어만).

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

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    result = response["message"]["content"].strip()
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