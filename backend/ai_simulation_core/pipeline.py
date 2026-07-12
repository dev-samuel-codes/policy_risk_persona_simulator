import json

from backend.ai_simulation_core.personas.persona_sampler import (
    get_citizen_persona,
    get_civil_servant_persona,
)
from backend.ai_simulation_core.policies.policy_repository import (
    get_random_policy,
    load_policies,
)
from backend.ai_simulation_core.simulations.citizen_simulation import (
    run_citizen_simulation,
)
from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    run_civil_servant_simulation,
)
from backend.scoring.risk_classifier import add_risk_categories
from backend.scoring.risk_scorer import compute_index, load_risk_pack


def run_pipeline() -> None:
    civil_personas = get_civil_servant_persona(
        limit=3, keyword="공무원", min_age=20, max_age=60
    )
    citizen_personas = get_citizen_persona(limit=3, excluded_keyword="공무원")
    policy = get_random_policy(load_policies())

    print("제시된 정책:", policy["상세정보"].get("서비스명"))

    simulation_results = []

    for index, persona in enumerate(citizen_personas, start=1):
        # 시민 응답 생성, JSON 파싱, 결과 검증을 한 번에 수행
        citizen_result = run_citizen_simulation(persona=persona, policy=policy)
        if citizen_result is None:
            print(f"\n===== 시민 PERSONA {index} 생성 실패 =====")
            continue

        simulation_results.append(citizen_result)

        # 같은 순번의 공무원 페르소나에게 현재 시민의 모든 민원을 전달
        civil_response = run_civil_servant_simulation(
            persona=civil_personas[index - 1],
            policy=policy,
            citizen_result=citizen_result,
        )

        print(f"\n===== 시민 PERSONA {index} RESULT =====")
        print(json.dumps(citizen_result, ensure_ascii=False, indent=2))

        print(f"\n===== 공무원 PERSONA {index} RESULT =====")
        print(civil_response)

    if not simulation_results:
        print("\n정책 민원 리스크를 계산할 시민 응답이 없습니다.")
        return

    # 모든 시민 응답 생성이 끝난 뒤 정책 전체 리스크를 한 번만 계산
    classified_results = add_risk_categories(simulation_results)
    risk_score = compute_index(classified_results, load_risk_pack())

    print("\n===== 정책 민원 리스크 =====")
    print(json.dumps(risk_score, ensure_ascii=False, indent=2))
