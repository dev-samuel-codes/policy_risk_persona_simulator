import json

from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import run_llm
from backend.ai_simulation_core.llm_inference.personas_filter import (
    get_civil_servant_persona,
    get_citizen_persona,
)
from backend.ai_simulation_core.llm_inference.prompts.civil_servant import (
    civil_servant_prompt,
)
from backend.ai_simulation_core.llm_inference.prompts.citizen import citizen_prompt
from backend.ai_simulation_core.llm_inference.get_poilcy_data import (
    load_policies,
    get_random_policy,
)
from backend.scoring.classifier import add_risk_categories
from backend.scoring.scorer import compute_index, load_risk_pack


def main() -> None:
    civil_personas = get_civil_servant_persona(
        limit=3, keyword="공무원", min_age=20, max_age=60
    )
    citizen_personas = get_citizen_persona(limit=3, excluded_keyword="공무원")
    policies = load_policies()

    policy = get_random_policy(policies)

    print("제시된 정책:", policy["상세정보"].get("서비스명"))

    simulation_results = []

    for index, persona in enumerate(citizen_personas, start=1):
        citizen_prompt_text = citizen_prompt(persona=persona, policy=policy)

        citizen_response = run_llm(citizen_prompt_text)
        citizen_result = json.loads(citizen_response)
        simulation_results.append(citizen_result)

        civil_persona = civil_personas[index - 1]
        citizen_complaint = "\n".join(
            complaint["dialogue"] for complaint in citizen_result["complaints"]
        )

        civil_prompt_text = civil_servant_prompt(
            persona=civil_persona,
            policy=policy,
            citizen_complaint=citizen_complaint,
        )

        civil_response = run_llm(civil_prompt_text)
        risk_pack = load_risk_pack()
        classified_results = add_risk_categories(simulation_results)
        risk_score = compute_index(classified_results, risk_pack)

        print(f"\n===== 시민 PERSONA {index} RESULT =====")
        print(citizen_response)

        print(f"\n===== 공무원 PERSONA {index} RESULT =====")
        print(civil_response)

        print("\n===== 정책 민원 리스크 =====")
        print(json.dumps(risk_score, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
