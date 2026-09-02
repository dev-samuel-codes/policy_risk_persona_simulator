import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.ai_simulation_core.pipeline import run_pipeline
from backend.ai_simulation_core.personas.persona_catalog import (
    get_persona_candidates,
    get_region_options,
    resolve_personas,
    validate_persona_selection,
)
from backend.ai_simulation_core.policies.policy_file_extractor import (
    PolicyFileExtractionError,
    extract_policy_fields_from_file,
)
from backend.ai_simulation_core.policies.policy_repository import (
    build_direct_policy,
    load_active_policy,
    save_active_policy,
)
from backend.ai_simulation_core.policies.policy_similarity import (
    DEFAULT_MIN_SCORE,
    PolicyIndexUnavailableError,
    find_similar_policies,
)
from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    validate_civil_servant_response,
)

app = FastAPI(title="CivicEcho API", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = PROJECT_ROOT / "data" / "runtime" / "simulations"
simulation_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="policy-simulation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def simulation_job_path(job_id: str) -> Path:
    try:
        normalized_job_id = str(UUID(job_id))
    except ValueError as error:
        raise ValueError("시뮬레이션 작업 ID가 올바르지 않습니다.") from error
    return SIMULATION_DIR / f"{normalized_job_id}.json"


def write_simulation_job(job: dict) -> None:
    job_path = simulation_job_path(job["job_id"])
    job_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = job_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(job, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(job_path)


def remove_deprecated_scoring_fields(job: dict) -> dict:
    """과거 저장 작업의 폐기된 점수 필드를 API 결과에서 제거한다."""

    result = job.get("result")
    if not isinstance(result, dict):
        return job

    result.pop("risk_score", None)
    citizen_results = result.get("citizen_results")
    if not isinstance(citizen_results, list):
        return job

    for citizen_result in citizen_results:
        if not isinstance(citizen_result, dict):
            continue
        complaints = citizen_result.get("complaints")
        if not isinstance(complaints, list):
            continue
        for complaint in complaints:
            if isinstance(complaint, dict):
                complaint.pop("risk_category", None)
    return job


def load_simulation_job(job_id: str) -> dict | None:
    job_path = simulation_job_path(job_id)
    if not job_path.exists():
        return None
    job = json.loads(job_path.read_text(encoding="utf-8"))
    return remove_deprecated_scoring_fields(job)


def validate_completed_simulation_result(
    result: object,
    expected_persona_ids: object,
) -> None:
    """API가 완료 상태를 저장하기 전에 3인 결과의 무결성을 확인한다."""

    if not isinstance(expected_persona_ids, list) or not expected_persona_ids:
        # 과거 작업 파일과 내부 호출은 persona_ids가 없을 수 있다.
        return
    expected_ids = [str(persona_id) for persona_id in expected_persona_ids]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("선택된 시민 페르소나 ID에 중복이 있습니다.")
    if not isinstance(result, dict):
        raise RuntimeError("시뮬레이션 완료 결과가 JSON 객체가 아닙니다.")

    citizen_results = result.get("citizen_results")
    civil_servant_results = result.get("civil_servant_results")
    if not isinstance(citizen_results, list) or len(citizen_results) != len(
        expected_ids
    ):
        raise RuntimeError(
            "시민 응답 수가 선택된 페르소나 수와 일치하지 않습니다."
        )
    if not isinstance(civil_servant_results, list) or len(
        civil_servant_results
    ) != len(expected_ids):
        raise RuntimeError(
            "공무원 응답 수가 선택된 페르소나 수와 일치하지 않습니다."
        )

    actual_ids = []
    for index, citizen_result in enumerate(citizen_results):
        if not isinstance(citizen_result, dict):
            raise RuntimeError(f"시민 응답 {index + 1}이 JSON 객체가 아닙니다.")
        actual_id = str(citizen_result.get("persona_id") or "")
        actual_ids.append(actual_id)
        persona = citizen_result.get("persona")
        nested_id = (
            str(persona.get("uuid") or "") if isinstance(persona, dict) else ""
        )
        if nested_id != actual_id:
            raise RuntimeError(
                f"시민 응답 {index + 1}의 페르소나 연결이 일치하지 않습니다."
            )
        if citizen_result.get("_validation_errors") != []:
            raise RuntimeError(
                f"시민 응답 {index + 1}에 검증 오류가 남아 있습니다."
            )
        quality_gate = citizen_result.get("_quality_gate")
        if not isinstance(quality_gate, dict) or quality_gate.get("status") != "passed":
            raise RuntimeError(
                f"시민 응답 {index + 1}이 의미 품질 검증을 통과하지 않았습니다."
            )

    if actual_ids != expected_ids:
        raise RuntimeError(
            "완료 결과의 시민 페르소나 ID 또는 순서가 요청과 다릅니다."
        )

    policy = result.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("완료 결과에 검증 가능한 정책 객체가 없습니다.")
    for index, (official_result, citizen_result) in enumerate(
        zip(civil_servant_results, citizen_results, strict=True),
        start=1,
    ):
        if not isinstance(official_result, dict):
            raise RuntimeError(f"공무원 응답 {index}가 JSON 객체가 아닙니다.")
        if official_result.get("persona_index") != index:
            raise RuntimeError(f"공무원 응답 {index}의 순번 연결이 일치하지 않습니다.")
        persona = official_result.get("persona")
        if not isinstance(persona, dict):
            raise RuntimeError(f"공무원 응답 {index}의 페르소나가 없습니다.")
        errors = validate_civil_servant_response(
            official_result,
            persona=persona,
            policy=policy,
            citizen_result=citizen_result,
            enforce_content_quality=False,
        )
        if errors:
            raise RuntimeError(
                f"공무원 응답 {index}의 연결 구조가 올바르지 않습니다: "
                + ", ".join(errors)
            )

def run_simulation_job(
    job_id: str,
    policy: dict,
    citizen_personas: list[dict] | None = None,
) -> None:
    queued_job = load_simulation_job(job_id)
    if queued_job is None:
        raise ValueError("시뮬레이션 작업이 없습니다.")

    created_at = queued_job["created_at"]
    similar_policies = queued_job.get("similar_policies", [])
    similarity = queued_job.get("similarity")
    started_at = utc_now()
    running_job = {
        **queued_job,
        "job_id": job_id,
        "status": "running",
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": None,
        "policy": policy,
        "similar_policies": similar_policies,
        "similarity": similarity,
        "result": None,
        "error": None,
    }
    write_simulation_job(running_job)

    try:
        if citizen_personas is None:
            result = run_pipeline(policy=policy)
        else:
            result = run_pipeline(
                policy=policy,
                citizen_personas=citizen_personas,
            )
        validate_completed_simulation_result(
            result,
            queued_job.get("persona_ids"),
        )
    except Exception as error:
        write_simulation_job(
            {
                **running_job,
                "job_id": job_id,
                "status": "failed",
                "created_at": created_at,
                "started_at": started_at,
                "completed_at": utc_now(),
                "policy": policy,
                "similar_policies": similar_policies,
                "similarity": similarity,
                "result": None,
                "error": str(error),
            }
        )
        return

    write_simulation_job(
        {
            **running_job,
            "job_id": job_id,
            "status": "completed",
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": utc_now(),
            "policy": policy,
            "similar_policies": similar_policies,
            "similarity": similarity,
            "result": result,
            "error": None,
        }
    )


class DirectPolicyInput(BaseModel):
    policy_name: str
    target_audience: str = ""
    selection_criteria: str = ""
    application_period: str = ""
    effective_date: str = ""
    required_documents: str = ""
    application_method: str = ""
    contact: str = ""
    benefits: str
    exclusion_conditions: str = ""
    region_scope: Literal["nationwide", "specific"] = "nationwide"
    region_province: str = ""
    region_district: str = ""
    age_min: int | None = Field(default=None, ge=0, le=120)
    age_max: int | None = Field(default=None, ge=0, le=120)
    age_basis: Literal["dataset_age"] = "dataset_age"

    @field_validator(
        "policy_name",
        "target_audience",
        "selection_criteria",
        "application_period",
        "effective_date",
        "required_documents",
        "application_method",
        "contact",
        "benefits",
        "exclusion_conditions",
        "region_province",
        "region_district",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("policy_name", "benefits")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("필수 입력값입니다.")
        return value

    @model_validator(mode="after")
    def validate_selection_scope(self) -> "DirectPolicyInput":
        if self.region_scope == "nationwide" and (
            self.region_province or self.region_district
        ):
            raise ValueError(
                "전국 정책에는 region_province와 region_district를 지정할 수 없습니다."
            )
        if self.region_scope == "specific" and not self.region_province:
            raise ValueError("특정 지역 정책에는 region_province가 필요합니다.")
        if (
            self.age_min is not None
            and self.age_max is not None
            and self.age_min > self.age_max
        ):
            raise ValueError("age_min은 age_max보다 클 수 없습니다.")
        return self


class PersonaSimulationInput(BaseModel):
    policy: DirectPolicyInput
    selection_mode: Literal["manual", "random"] = "manual"
    persona_ids: list[str] = Field(default_factory=list, max_length=3)
    selection_cohorts: list[
        Literal["eligible", "boundary", "region_boundary"]
    ] = Field(default_factory=list, max_length=3)

    @field_validator("persona_ids")
    @classmethod
    def validate_persona_ids(cls, values: list[str]) -> list[str]:
        normalized = [str(value or "").strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("페르소나 ID는 비어 있을 수 없습니다.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("서로 다른 페르소나 3명을 선택해야 합니다.")
        return normalized

    @model_validator(mode="after")
    def validate_selection_mode(self) -> "PersonaSimulationInput":
        if self.selection_mode == "manual" and len(self.persona_ids) != 3:
            raise ValueError("수동 선택에서는 서로 다른 페르소나 3명이 필요합니다.")
        if (
            self.selection_mode == "manual"
            and self.selection_cohorts
            and len(self.selection_cohorts) != len(self.persona_ids)
        ):
            raise ValueError("페르소나 ID와 후보 유형 수가 일치해야 합니다.")
        if self.selection_mode == "random" and self.persona_ids:
            raise ValueError(
                "무작위 선택에서는 persona_ids를 함께 지정할 수 없습니다."
            )
        if self.selection_mode == "random" and self.selection_cohorts:
            raise ValueError(
                "무작위 선택에서는 selection_cohorts를 함께 지정할 수 없습니다."
            )
        return self


class SimilarPolicyInput(DirectPolicyInput):
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0, le=1)


def policy_similarity_metadata(similarity: dict) -> dict:
    return {
        key: similarity[key]
        for key in (
            "as_of_date",
            "index_version",
            "source_count",
            "source",
            "query_time_ms",
        )
        if key in similarity
    }


def search_similar_policies(
    policy: dict, *, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE
) -> dict:
    try:
        return find_similar_policies(
            policy,
            top_k=top_k,
            min_score=min_score,
        )
    except PolicyIndexUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/personas/options")
def get_persona_options(response: Response) -> dict:
    try:
        options = get_region_options()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    serialized = json.dumps(
        options,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["ETag"] = f'"{sha256(serialized).hexdigest()}"'
    return options


@app.get("/api/personas/candidates")
def get_persona_candidate_options(
    region_scope: Literal["nationwide", "specific"] = Query(default="nationwide"),
    province: str = Query(default=""),
    district: str = Query(default=""),
    age_min: int | None = Query(default=None, ge=0, le=120),
    age_max: int | None = Query(default=None, ge=0, le=120),
    cohort: Literal["eligible", "boundary", "region_boundary"] = Query(
        default="eligible"
    ),
    limit: int = Query(default=12, ge=1, le=24),
    seed: int = Query(default=0),
) -> dict:
    normalized_province = province.strip()
    normalized_district = district.strip()
    try:
        candidates = get_persona_candidates(
            region_scope=region_scope,
            province=normalized_province,
            district=normalized_district,
            age_min=age_min,
            age_max=age_max,
            cohort=cohort,
            limit=limit,
            seed=seed,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "region_scope": region_scope,
        "province": normalized_province,
        "district": normalized_district,
        "age_min": age_min,
        "age_max": age_max,
        "cohort": cohort,
        "seed": seed,
        "candidates": candidates,
    }


@app.get("/api/policies/active")
def get_active_policy() -> dict:
    policy = load_active_policy()
    if policy is None:
        raise HTTPException(status_code=404, detail="활성 정책이 없습니다.")
    return {"status": "ok", "policy": policy}


@app.post("/api/policies/extract-file")
async def extract_policy_file(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        fields = extract_policy_fields_from_file(file.filename or "", content)
    except PolicyFileExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {"status": "ok", "fields": fields}


@app.post("/api/policies/similar")
def get_similar_policies(payload: SimilarPolicyInput) -> dict:
    try:
        policy = build_direct_policy(payload.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    similarity = search_similar_policies(
        policy,
        top_k=payload.top_k,
        min_score=payload.min_score,
    )
    return {
        "status": "ok",
        "policy": policy,
        **similarity,
    }


@app.post("/api/policies/direct", deprecated=True)
def set_direct_policy(payload: DirectPolicyInput) -> dict:
    """기존 호출자를 유지하되 필터가 적용되는 무작위 선택 경로로 위임한다."""

    return create_persona_simulation(
        PersonaSimulationInput(
            policy=payload,
            selection_mode="random",
        )
    )


@app.post("/api/simulations")
def create_persona_simulation(payload: PersonaSimulationInput) -> dict:
    selection_seed: int | None = None
    try:
        policy = build_direct_policy(payload.policy.model_dump())
        if payload.selection_mode == "random":
            selection_seed = secrets.randbits(63)
            sampled_candidates = get_persona_candidates(
                region_scope=payload.policy.region_scope,
                province=payload.policy.region_province,
                district=payload.policy.region_district,
                age_min=payload.policy.age_min,
                age_max=payload.policy.age_max,
                cohort="eligible",
                limit=3,
                seed=selection_seed,
            )
            if len(sampled_candidates) != 3:
                raise ValueError(
                    "현재 지역·나이 범위에서 분석 가능한 페르소나 3명을 "
                    "찾지 못했습니다."
                )
            citizen_personas = [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "match"
                }
                for candidate in sampled_candidates
            ]
        else:
            citizen_personas = resolve_personas(payload.persona_ids)

        requested_selection_cohorts = (
            ["eligible"] * len(citizen_personas)
            if payload.selection_mode == "random"
            else list(payload.selection_cohorts) or None
        )
        selection_match = validate_persona_selection(
            citizen_personas,
            region_scope=payload.policy.region_scope,
            province=payload.policy.region_province,
            district=payload.policy.region_district,
            age_min=payload.policy.age_min,
            age_max=payload.policy.age_max,
            selection_cohorts=requested_selection_cohorts,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    similarity = search_similar_policies(policy)
    save_active_policy(policy)
    similar_policies = similarity["results"]
    similarity_metadata = policy_similarity_metadata(similarity)
    selected_personas = [
        {
            **persona,
            "selection_cohort": match["selection_cohort"],
            "selection_match": match,
        }
        for persona, match in zip(
            citizen_personas,
            selection_match,
            strict=True,
        )
    ]
    selected_cohorts = [match["selection_cohort"] for match in selection_match]
    selected_persona_ids = [str(persona["uuid"]) for persona in citizen_personas]

    job_id = str(uuid4())
    write_simulation_job(
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "policy": policy,
            "selection_mode": payload.selection_mode,
            "selection_seed": selection_seed,
            "persona_ids": selected_persona_ids,
            "selection_cohorts": selected_cohorts,
            "selected_personas": selected_personas,
            "selection_match": selection_match,
            "similar_policies": similar_policies,
            "similarity": similarity_metadata,
            "result": None,
            "error": None,
        }
    )
    simulation_executor.submit(
        run_simulation_job,
        job_id,
        policy,
        citizen_personas,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "source": "direct_input",
        "policy": policy,
        "selection_mode": payload.selection_mode,
        "selection_seed": selection_seed,
        "persona_ids": selected_persona_ids,
        "selection_cohorts": selected_cohorts,
        "selected_personas": selected_personas,
        "selection_match": selection_match,
        "similar_policies": similar_policies,
        "similarity": similarity_metadata,
    }


@app.get("/api/simulations/{job_id}")
def get_simulation_job(job_id: str) -> dict:
    try:
        job = load_simulation_job(job_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if job is None:
        raise HTTPException(status_code=404, detail="시뮬레이션 작업이 없습니다.")
    return job
