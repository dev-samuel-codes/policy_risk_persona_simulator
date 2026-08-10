import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from backend.ai_simulation_core.pipeline import run_pipeline
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


def load_simulation_job(job_id: str) -> dict | None:
    job_path = simulation_job_path(job_id)
    if not job_path.exists():
        return None
    return json.loads(job_path.read_text(encoding="utf-8"))


def run_simulation_job(job_id: str, policy: dict) -> None:
    queued_job = load_simulation_job(job_id)
    if queued_job is None:
        raise ValueError("시뮬레이션 작업이 없습니다.")

    created_at = queued_job["created_at"]
    similar_policies = queued_job.get("similar_policies", [])
    similarity = queued_job.get("similarity")
    started_at = utc_now()
    write_simulation_job(
        {
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
    )

    try:
        result = run_pipeline(policy=policy)
    except Exception as error:
        write_simulation_job(
            {
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
    application_period: str = ""
    effective_date: str = ""
    required_documents: str = ""
    application_method: str = ""
    contact: str = ""
    benefits: str
    exclusion_conditions: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("policy_name", "benefits")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("필수 입력값입니다.")
        return value


class SimilarPolicyInput(DirectPolicyInput):
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0, le=1)


def policy_similarity_metadata(similarity: dict) -> dict:
    return {
        key: similarity[key]
        for key in ("as_of_date", "index_version", "source_count", "query_time_ms")
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


@app.post("/api/policies/direct")
def set_direct_policy(payload: DirectPolicyInput) -> dict:
    try:
        policy = build_direct_policy(payload.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    similarity = search_similar_policies(policy)
    save_active_policy(policy)
    similar_policies = similarity["results"]
    similarity_metadata = policy_similarity_metadata(similarity)

    job_id = str(uuid4())
    write_simulation_job(
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "policy": policy,
            "similar_policies": similar_policies,
            "similarity": similarity_metadata,
            "result": None,
            "error": None,
        }
    )
    simulation_executor.submit(run_simulation_job, job_id, policy)

    return {
        "status": "queued",
        "job_id": job_id,
        "source": "direct_input",
        "policy": policy,
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
