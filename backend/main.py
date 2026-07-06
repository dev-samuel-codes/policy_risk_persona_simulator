# 백엔드 메인 실행점
from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_llm
from backend.ai_simulation_core.llm_inference.personas_filter import get_persona

# 페르소나를 하나 받아서 LLM에게 보낼 프롬프트 문장을 생성
def build_prompt(persona: dict) -> str:
    return f"""
다음 페르소나를 기반으로 정책 리스크를 분석하세요.

페르소나:
{persona}
"""


def main() -> None:
    # 1. 네모트론 페르소나 데이터가 없으면 다운로드하고, 있으면 스킵
    # get_persona() 내부에서 download_dataset()을 호출함

    # 2. 사용할 페르소나 가져오기
    personas = get_persona(limit=3, keyword="공무원")

    # 3. 페르소나별로 프롬프트 생성 후 LLM 실행
    for index, persona in enumerate(personas, start=1):
        prompt = build_prompt(persona)
        response = run_llm(prompt)

        print(f"\n===== PERSONA {index} RESULT =====")
        print(response)


if __name__ == "__main__":
    main()