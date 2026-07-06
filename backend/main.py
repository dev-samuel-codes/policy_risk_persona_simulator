from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_llm
from backend.ai_simulation_core.llm_inference.personas_filter import get_persona
from backend.ai_simulation_core.llm_inference.prompts.civil_servant import (
    civil_servant_prompt,
)


def main() -> None:
    personas = get_persona(limit=3, keyword="공무원", min_age=20, max_age=60)

    for index, persona in enumerate(personas, start=1):
        prompt = civil_servant_prompt(persona)
        response = run_llm(prompt)

        print(f"\n===== PERSONA {index} RESULT =====")
        print(response)


if __name__ == "__main__":
    main()