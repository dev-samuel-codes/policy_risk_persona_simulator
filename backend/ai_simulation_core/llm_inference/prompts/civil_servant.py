# 공무원 전용 프롬프트

from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_llm
from backend.ai_simulation_core.llm_inference.personas_filter import get_persona

def civil_servant_prompt(persona: dict) -> str:
    persona_text = "\n".join(
        f"{key}: {value}"
        for key, value in persona.items()
        if value is not None
    )

    return f"""
다음 공무원 페르소나를 기반으로 정책 리스크를 분석하세요.
반드시 한국어로만 답하세요.
추론 과정은 출력하지 말고 최종 분석만 출력하세요.

분석할 때 다음 관점을 반영하세요.
1. 공무원으로서의 정책 집행 리스크
2. 민원 발생 가능성
3. 이해관계자 갈등 가능성
4. 지역 특성과 생활 패턴이 정책 수용성에 미치는 영향
5. 대응 전략

출력 형식:
1. 이름과: [] / 직업: [] / 성별: [] / 나이: []
2. 정책 리스크 요약: []
3. 예상 민원 또는 갈등 지점: []
4. 정책 집행 시 주의할 점: []
5. 대응 전략: []

[]를 무조건 채워서 출력하세요

페르소나:
{persona_text}
""".strip()