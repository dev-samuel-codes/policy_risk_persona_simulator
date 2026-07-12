# 시민에게 넣을 정책 가저오기

import json
import random
from pathlib import Path

# 현재 파일 위치 기준으로 정책 데이터 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "policies"


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
