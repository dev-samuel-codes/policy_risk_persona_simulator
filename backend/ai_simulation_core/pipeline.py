import json

from backend.ai_simulation_core.complaints.civil_complaint_similarity import (
    CivilComplaintIndexUnavailableError,
    find_similar_complaint_cases_batch,
)
from backend.ai_simulation_core.llm.llm_gateway import unload_llm
from backend.ai_simulation_core.personas.persona_sampler import (
    get_citizen_persona,
    get_civil_servant_persona,
)
from backend.ai_simulation_core.policies.policy_repository import (
    get_active_or_random_policy,
)
from backend.ai_simulation_core.simulations.citizen_simulation import (
    run_citizen_simulation,
)
from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    run_civil_servant_simulation,
    validate_civil_servant_response,
)

def _complaint_query(complaint: dict) -> str:
    # The dialogue is the actual citizen-facing message shown prominently in the
    # UI. The shorter complaint_text summary is only a fallback because generic
    # summaries can over-match unrelated FAQ entries that also mention support.
    for field in ("dialogue", "complaint_text"):
        value = complaint.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _mark_reference_search_unavailable(
    complaints: list[dict],
    *,
    reason_code: str,
) -> None:
    for complaint in complaints:
        complaint["reference_cases"] = []
        complaint["precedent_search"] = {
            "status": "unavailable",
            "reason_code": reason_code,
        }


def _valid_reference_search_contract(status: object, reference_cases: object) -> bool:
    if status not in {"matched", "no_reliable_match", "invalid_query"}:
        return False
    if not isinstance(reference_cases, list) or not all(
        isinstance(reference_case, dict) for reference_case in reference_cases
    ):
        return False
    if status == "matched":
        return any(
            reference_case.get("reference_eligible") is True
            for reference_case in reference_cases
        )
    return not reference_cases


def attach_complaint_reference_cases(simulation_results: list[dict]) -> None:
    searchable_complaints = []
    batch_items = []

    for citizen_result in simulation_results:
        complaints = citizen_result.get("complaints", [])
        if not isinstance(complaints, list):
            continue

        for complaint in complaints:
            if not isinstance(complaint, dict):
                continue

            complaint_text = _complaint_query(complaint)
            if not complaint_text:
                complaint["reference_cases"] = []
                complaint["precedent_search"] = {
                    "status": "invalid_query",
                    "reason_code": "empty_complaint_text",
                }
                continue

            searchable_complaints.append(complaint)
            batch_items.append({"complaint_text": complaint_text})

    try:
        search_results = find_similar_complaint_cases_batch(batch_items)
    except CivilComplaintIndexUnavailableError:
        _mark_reference_search_unavailable(
            searchable_complaints,
            reason_code="index_unavailable",
        )
        return
    except Exception:
        # 참고 민원 검색은 보조 기능이므로 검색 장애가 시뮬레이션을 중단시키지 않는다.
        _mark_reference_search_unavailable(
            searchable_complaints,
            reason_code="search_failed",
        )
        return

    if not isinstance(search_results, list) or len(search_results) != len(
        searchable_complaints
    ):
        _mark_reference_search_unavailable(
            searchable_complaints,
            reason_code="invalid_search_response",
        )
        return

    for complaint, search_result in zip(
        searchable_complaints,
        search_results,
        strict=True,
    ):
        if not isinstance(search_result, dict):
            _mark_reference_search_unavailable(
                [complaint],
                reason_code="invalid_search_response",
            )
            continue

        status = search_result.get("status")
        reference_cases = search_result.get("results")
        if not _valid_reference_search_contract(status, reference_cases):
            _mark_reference_search_unavailable(
                [complaint],
                reason_code="invalid_search_response",
            )
            continue

        complaint["reference_cases"] = reference_cases
        complaint["precedent_search"] = {
            key: value for key, value in search_result.items() if key != "results"
        }


def compute_complaint_reference_summary(
    simulation_results: list[dict],
) -> dict:
    total = 0
    matched = 0
    no_reliable_match = 0
    unavailable = 0
    invalid = 0

    for citizen_result in simulation_results:
        complaints = citizen_result.get("complaints", [])
        if not isinstance(complaints, list):
            continue

        for complaint in complaints:
            if not isinstance(complaint, dict):
                continue
            total += 1
            precedent_search = complaint.get("precedent_search", {})
            status = (
                precedent_search.get("status")
                if isinstance(precedent_search, dict)
                else None
            )
            if status == "matched":
                matched += 1
            elif status == "no_reliable_match":
                no_reliable_match += 1
            elif status == "invalid_query":
                invalid += 1
            else:
                unavailable += 1

    evaluated = matched + no_reliable_match
    reference_rate = round(matched / evaluated * 100, 1) if evaluated > 0 else None
    search_coverage = round(evaluated / total * 100, 1) if total > 0 else 0.0

    if unavailable == 0:
        status = "available"
    elif evaluated > 0:
        status = "partial"
    else:
        status = "unavailable"

    return {
        "total": total,
        "evaluated": evaluated,
        "matched": matched,
        "unavailable": unavailable,
        "invalid": invalid,
        "reference_rate": reference_rate,
        "search_coverage": search_coverage,
        "status": status,
    }


def run_pipeline(
    policy: dict | None = None,
    citizen_personas: list[dict] | None = None,
) -> dict:
    try:
        # 기존 무인자 호출은 내부 호출 형태까지 보존한다. 일부 실행 래퍼와
        # 수명주기 검증은 이 경로를 기준으로 동작한다.
        if policy is None and citizen_personas is None:
            return _run_pipeline()
        return _run_pipeline(
            policy=policy,
            citizen_personas=citizen_personas,
        )
    finally:
        # 정상 완료, 조기 반환, 예외 및 Ctrl+C 모두에서 LLM 프로세스를 종료한다.
        unload_llm()


def _run_pipeline(
    policy: dict | None = None,
    citizen_personas: list[dict] | None = None,
) -> dict:
    citizen_count = len(citizen_personas) if citizen_personas is not None else 3
    if citizen_count <= 0:
        raise ValueError("실행할 시민 페르소나가 없습니다.")

    civil_personas = get_civil_servant_persona(
        limit=citizen_count,
        keyword="공무원",
        min_age=20,
        max_age=60,
    )
    if citizen_personas is None:
        # 기존 무작위 실행은 종전과 동일하게 공무원 추출 뒤 시민을 추출한다.
        citizen_personas = get_citizen_persona(
            limit=3,
            excluded_keyword="공무원",
        )
    if len(citizen_personas) != citizen_count:
        raise RuntimeError(
            f"요청한 시민 페르소나 수와 실제 추출 수가 다릅니다: "
            f"요청={citizen_count}, 추출={len(citizen_personas)}"
        )
    if len(civil_personas) < citizen_count:
        raise RuntimeError(
            f"시민 응답에 필요한 공무원 페르소나가 부족합니다: "
            f"필요={citizen_count}, 추출={len(civil_personas)}"
        )
    selected_policy = policy or get_active_or_random_policy()

    print("제시된 정책:", selected_policy["상세정보"].get("서비스명"))

    simulation_results = []
    civil_servant_results = []

    for index, persona in enumerate(citizen_personas, start=1):
        # 시민 응답 생성, JSON 파싱, 결과 검증을 한 번에 수행
        citizen_result = run_citizen_simulation(persona=persona, policy=selected_policy)
        if citizen_result is None:
            persona_id = persona.get("uuid", f"index-{index}")
            raise RuntimeError(
                f"시민 페르소나 응답이 품질 검증을 통과하지 못했습니다: "
                f"{persona_id}"
            )

        # 결과 화면에서 생성 대사와 실제 샘플링 페르소나를 연결할 수 있도록 보존
        citizen_result["persona"] = persona
        simulation_results.append(citizen_result)

        # 같은 순번의 공무원 페르소나에게 현재 시민의 모든 민원을 전달
        civil_persona = civil_personas[index - 1]
        civil_response = run_civil_servant_simulation(
            persona=civil_persona,
            policy=selected_policy,
            citizen_result=citizen_result,
        )
        civil_errors = validate_civil_servant_response(
            civil_response,
            persona=civil_persona,
            policy=selected_policy,
            citizen_result=citizen_result,
            enforce_content_quality=False,
        )
        if civil_errors:
            persona_id = civil_persona.get("uuid", f"index-{index}")
            raise RuntimeError(
                "공무원 페르소나 응답의 연결 구조가 올바르지 않습니다: "
                f"{persona_id} ({', '.join(civil_errors)})"
            )
        civil_servant_results.append(
            {
                "persona_index": index,
                "persona": civil_persona,
                **civil_response,
            }
        )

        print(f"\n===== 시민 PERSONA {index} RESULT =====")
        print(json.dumps(citizen_result, ensure_ascii=False, indent=2))

        print(f"\n===== 공무원 PERSONA {index} RESULT =====")
        print(civil_response["response"])

    # 모든 생성이 끝난 뒤 한 번의 배치 검색으로 민원 순서와 검색 결과를 연결한다.
    attach_complaint_reference_cases(simulation_results)
    complaint_reference_summary = compute_complaint_reference_summary(
        simulation_results
    )

    if len(simulation_results) != citizen_count:
        raise RuntimeError(
            f"품질 검증을 통과한 시민 응답 수가 요청 수와 다릅니다: "
            f"요청={citizen_count}, 결과={len(simulation_results)}"
        )

    return {
        "policy": selected_policy,
        "citizen_results": simulation_results,
        "civil_servant_results": civil_servant_results,
        "complaint_reference_summary": complaint_reference_summary,
    }
