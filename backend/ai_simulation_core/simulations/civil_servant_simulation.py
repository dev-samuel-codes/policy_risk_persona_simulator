from backend.ai_simulation_core.llm.llm_gateway import run_llm
from backend.ai_simulation_core.prompts.civil_servant_prompt import (
    civil_servant_prompt,
)


def run_civil_servant_simulation(
    persona: dict, policy: dict, citizen_result: dict
) -> str:
    citizen_complaint = "\n".join(
        complaint["dialogue"] for complaint in citizen_result["complaints"]
    )
    prompt = civil_servant_prompt(
        persona=persona,
        policy=policy,
        citizen_complaint=citizen_complaint,
    )
    return run_llm(prompt)
