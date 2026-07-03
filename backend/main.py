from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_qwen
from backend.ai_simulation_core.llm_inference.nemotron_local_inference.get_nemotron_persona import (
    get_persona,
)

def build_prompt(persona: dict) -> str:
    return f"""
다음 페르소나를 기반으로 정책 리스크를 분석하세요.

페르소나:
{persona}
"""


def main() -> None:
    personas = get_persona(limit=3)

    for index, persona in enumerate(personas, start=1):
        prompt = build_prompt(persona)
        response = run_qwen(prompt)

        print(f"\n===== PERSONA {index} RESULT =====")
        print(response)


if __name__ == "__main__":
    main()
