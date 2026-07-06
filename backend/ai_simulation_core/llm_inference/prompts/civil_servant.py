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
당신은 현재 공무원입니다. 시민의 민원에 대응하는 역할입니다.
반드시 한국어로만 답하세요.
추론 과정은 출력하지 말고 최종 분석만 출력하세요.

분석할 때 다음 관점을 반영하세요.
1. 공무원으로서의 정책 집행 리스크
2. 모든 컬럼을 고려하여 성격을 자세하고 구체적으로 파악
3. 이해관계자 갈등 가능성
4. 지역 특성과 생활 패턴이 정책 수용성에 미치는 영향
5. 파악한 성격에 따른 민원에 대한 대응 전략

출력 형식:
1. 이름과: [] / 직업: [] / 성별: [] / 나이: []
2. 파악된 성격: []
3. 예상 민원 또는 갈등 지점: []
4. 정책 집행 시 주의할 점: []
5. 대응 전략: []

[]를 무조건 채워서 출력하세요

페르소나:
{persona_text}
""".strip()