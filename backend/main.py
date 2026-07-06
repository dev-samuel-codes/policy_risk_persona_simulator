from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_llm
from backend.ai_simulation_core.llm_inference.personas_filter import (
    get_civil_servant_persona,
    get_citizen_persona
)
from backend.ai_simulation_core.llm_inference.prompts.civil_servant import civil_servant_prompt
from backend.ai_simulation_core.llm_inference.prompts.citizen import citizen_prompt

def main() -> None:
    civil_personas = get_civil_servant_persona(limit=3, keyword="공무원", min_age=20, max_age=60)
    citizen_personas = get_citizen_persona(limit=3, excluded_keyword="공무원")

    for index, persona in enumerate(civil_personas, start=1):
        prompt = civil_servant_prompt(persona)
        response = run_llm(prompt)

        print(f"\n===== 공무원 PERSONA {index} RESULT =====")
        print(response)

    for index, persona in enumerate(citizen_personas, start=1):
        prompt = citizen_prompt(persona)
        response = run_llm(prompt)

        print(f"\n===== 시민 PERSONA {index} RESULT =====")
        print(response)


if __name__ == "__main__":
    main()