import json
import random
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "policies"
ACTIVE_POLICY_PATH = PROJECT_ROOT / "data" / "runtime" / "active_policy.json"


# JSON 파일을 하나 읽어서 안의 data 값 반환
def load_json_data(file_name: str) -> list[dict]:
    path = POLICY_DATA_DIR / file_name
    payload = json.loads(path.read_text(encoding="utf-8"))  # 한글이 존재 -> utf-8

    return payload["data"]


def load_policies() -> list[dict]:
    service_list = load_json_data("service_list.json")
    service_detail = load_json_data("service_detail.json")
    support_conditions = load_json_data("support_conditions.json")

    # 상세 정보를 서비스id 기준 딕셔너리로 변환
    detail_by_id = {item["서비스ID"]: item for item in service_detail}

    conditions_by_id = {  # 지원조건
        item["서비스ID"]: item for item in support_conditions
    }

    # 최종 정책 리스트 생성
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


# 랜덤으로 정책 선택
def get_random_policy(policies: list[dict]) -> dict:
    return random.choice(policies)


def build_direct_policy(fields: Mapping[str, str]) -> dict:
    normalized = {key: str(value or "").strip() for key, value in fields.items()}
    policy_name = normalized.get("policy_name", "")
    benefits = normalized.get("benefits", "")

    if not policy_name:
        raise ValueError("정책명은 필수입니다.")
    if not benefits:
        raise ValueError("혜택은 필수입니다.")

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
    ACTIVE_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ACTIVE_POLICY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(ACTIVE_POLICY_PATH)


def load_active_policy() -> dict | None:
    if not ACTIVE_POLICY_PATH.exists():
        return None

    policy = json.loads(ACTIVE_POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or not isinstance(policy.get("상세정보"), dict):
        raise ValueError("활성 정책 파일 형식이 올바르지 않습니다.")
    return policy


def get_active_or_random_policy() -> dict:
    active_policy = load_active_policy()
    if active_policy is not None:
        return active_policy
    return get_random_policy(load_policies())
