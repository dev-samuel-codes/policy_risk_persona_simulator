import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "policies"
ACTIVE_POLICY_PATH = PROJECT_ROOT / "data" / "runtime" / "active_policy.json"


def load_json_data(file_name: str) -> list[dict]:
    """정책 원천 JSON의 data 배열을 읽는다."""
    path = POLICY_DATA_DIR / file_name
    payload = json.loads(path.read_text(encoding="utf-8"))

    return payload["data"]


def load_policies() -> list[dict]:
    """목록·상세·지원조건 원천을 서비스 ID 기준으로 결합한다."""
    service_list = load_json_data("service_list.json")
    service_detail = load_json_data("service_detail.json")
    support_conditions = load_json_data("support_conditions.json")

    detail_by_id = {item["서비스ID"]: item for item in service_detail}
    conditions_by_id = {
        item["서비스ID"]: item for item in support_conditions
    }

    policies = []

    for item in service_list:
        service_id = item["서비스ID"]

        policy = {
            "목록정보": item,
            "상세정보": detail_by_id.get(service_id, {}),
            "지원조건": conditions_by_id.get(service_id, {}),
        }

        policies.append(policy)

    return policies


def get_random_policy(policies: list[dict]) -> dict:
    """정책 목록에서 시뮬레이션 입력 하나를 무작위로 고른다."""
    return random.choice(policies)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name}은 정수여야 합니다.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}은 정수여야 합니다.") from error


def build_direct_policy(fields: Mapping[str, Any]) -> dict:
    """직접 입력값을 검증해 시뮬레이션 내부 정책 스키마로 변환한다.

    지역 범위와 연령 하한·상한 계약을 확인하고, 파이프라인이 소비하는
    목록정보·상세정보·지원조건 구조를 반환한다.
    """
    text_fields = {
        "policy_name",
        "target_audience",
        "application_period",
        "effective_date",
        "required_documents",
        "application_method",
        "contact",
        "benefits",
        "exclusion_conditions",
        "region_scope",
        "region_province",
        "region_district",
        "age_basis",
    }
    normalized = {
        key: str(value or "").strip()
        for key, value in fields.items()
        if key in text_fields
    }
    policy_name = normalized.get("policy_name", "")
    benefits = normalized.get("benefits", "")

    if not policy_name:
        raise ValueError("정책명은 필수입니다.")
    if not benefits:
        raise ValueError("혜택은 필수입니다.")

    region_scope = normalized.get("region_scope") or "nationwide"
    region_province = normalized.get("region_province", "")
    region_district = normalized.get("region_district", "")
    age_basis = normalized.get("age_basis") or "dataset_age"
    age_min = _optional_int(fields.get("age_min"), "age_min")
    age_max = _optional_int(fields.get("age_max"), "age_max")

    if region_scope not in {"nationwide", "specific"}:
        raise ValueError("region_scope는 nationwide 또는 specific이어야 합니다.")
    if region_scope == "nationwide" and (region_province or region_district):
        raise ValueError(
            "전국 정책에는 region_province와 region_district를 지정할 수 없습니다."
        )
    if region_scope == "specific" and not region_province:
        raise ValueError("특정 지역 정책에는 region_province가 필요합니다.")
    if age_min is not None and age_max is not None and age_min > age_max:
        raise ValueError("age_min은 age_max보다 클 수 없습니다.")
    if age_basis != "dataset_age":
        raise ValueError("현재 age_basis는 dataset_age만 지원합니다.")

    exclusion_conditions = normalized.get("exclusion_conditions", "")
    policy_detail = {
        "서비스ID": "direct-input",
        "서비스명": policy_name,
        "지원대상": normalized.get("target_audience", ""),
        "신청기한": normalized.get("application_period", ""),
        "시행일": normalized.get("effective_date", ""),
        "구비서류": normalized.get("required_documents", ""),
        "신청방법": normalized.get("application_method", ""),
        "문의처": normalized.get("contact", ""),
        "지원내용": benefits,
        "제외조건": exclusion_conditions,
        "선정기준": exclusion_conditions,
    }

    return {
        "입력출처": "직접입력",
        "region_scope": region_scope,
        "region_province": region_province,
        "region_district": region_district,
        "age_min": age_min,
        "age_max": age_max,
        "age_basis": age_basis,
        "목록정보": {
            "서비스ID": "direct-input",
            "서비스명": policy_name,
            "지원대상": policy_detail["지원대상"],
            "신청기한": policy_detail["신청기한"],
            "신청방법": policy_detail["신청방법"],
            "전화문의": policy_detail["문의처"],
            "지원내용": benefits,
        },
        "상세정보": policy_detail,
        "지원조건": {},
    }


def save_active_policy(policy: dict) -> None:
    """검토가 끝난 정책을 활성 정책 파일로 원자적으로 저장한다."""
    ACTIVE_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ACTIVE_POLICY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(ACTIVE_POLICY_PATH)


def load_active_policy() -> dict | None:
    """활성 정책을 읽고 파이프라인이 요구하는 기본 구조를 확인한다."""
    if not ACTIVE_POLICY_PATH.exists():
        return None

    policy = json.loads(ACTIVE_POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or not isinstance(policy.get("상세정보"), dict):
        raise ValueError("활성 정책 파일 형식이 올바르지 않습니다.")
    return policy


def get_active_or_random_policy() -> dict:
    """활성 정책이 없을 때만 원천 정책 중 하나를 대체 입력으로 사용한다."""
    active_policy = load_active_policy()
    if active_policy is not None:
        return active_policy
    return get_random_policy(load_policies())
