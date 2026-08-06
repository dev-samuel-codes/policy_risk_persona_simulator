import hashlib
import json
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "policies"

SEARCH_FIELDS = (
    "policy_name",
    "purpose",
    "category",
    "support_type",
    "target_audience",
    "selection_criteria",
    "benefits",
    "application_period",
    "application_method",
)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_date(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    for pattern, length in (("%Y%m%d%H%M%S", 14), ("%Y%m%d", 8), ("%Y-%m-%d", 10)):
        try:
            return datetime.strptime(text[:length], pattern).date().isoformat()
        except ValueError:
            continue
    return ""


def date_ordinal(value: object, *, default: date | None = None) -> int:
    normalized = normalize_date(value)
    if normalized:
        return int(normalized.replace("-", ""))
    if default is not None:
        return int(default.strftime("%Y%m%d"))
    return 0


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"정책 원천 파일 형식이 올바르지 않습니다: {path}")
    return rows


def _by_service_id(
    rows: list[dict[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        service_id = normalize_text(row.get("서비스ID"))
        if not service_id:
            raise ValueError(f"{source}에 서비스ID가 없는 정책이 있습니다.")
        if service_id in result:
            raise ValueError(f"{source}에 중복 서비스ID가 있습니다: {service_id}")
        result[service_id] = row
    return result


def _condition_summary(row: dict[str, Any]) -> str:
    values = []
    for key, value in sorted(row.items()):
        if key in {"서비스ID", "서비스명"}:
            continue
        normalized = normalize_text(value)
        if normalized:
            values.append(f"{key}: {normalized}")
    return " | ".join(values)


def normalize_policy_record(
    list_item: dict[str, Any],
    detail_item: dict[str, Any],
    condition_item: dict[str, Any],
) -> dict[str, Any]:
    service_id = normalize_text(
        list_item.get("서비스ID") or detail_item.get("서비스ID")
    )
    if not service_id:
        raise ValueError("정책에 서비스ID가 없습니다.")

    return {
        "service_id": service_id,
        "policy_name": normalize_text(
            detail_item.get("서비스명") or list_item.get("서비스명")
        ),
        "purpose": normalize_text(
            detail_item.get("서비스목적") or list_item.get("서비스목적요약")
        ),
        "category": normalize_text(list_item.get("서비스분야")),
        "support_type": normalize_text(
            detail_item.get("지원유형") or list_item.get("지원유형")
        ),
        "target_audience": normalize_text(
            detail_item.get("지원대상") or list_item.get("지원대상")
        ),
        "selection_criteria": normalize_text(
            detail_item.get("선정기준") or list_item.get("선정기준")
        ),
        "benefits": normalize_text(
            detail_item.get("지원내용") or list_item.get("지원내용")
        ),
        "application_period": normalize_text(
            detail_item.get("신청기한") or list_item.get("신청기한")
        ),
        "application_method": normalize_text(
            detail_item.get("신청방법") or list_item.get("신청방법")
        ),
        "required_documents": normalize_text(detail_item.get("구비서류")),
        "contact": normalize_text(
            detail_item.get("문의처") or list_item.get("전화문의")
        ),
        "organization": normalize_text(
            detail_item.get("소관기관명") or list_item.get("소관기관명")
        ),
        "organization_type": normalize_text(list_item.get("소관기관유형")),
        "registered_at": normalize_date(list_item.get("등록일시")),
        "modified_at": normalize_date(
            detail_item.get("수정일시") or list_item.get("수정일시")
        ),
        "source_url": normalize_text(
            detail_item.get("온라인신청사이트URL") or list_item.get("상세조회URL")
        ),
        "support_conditions": _condition_summary(condition_item),
    }


@lru_cache(maxsize=2)
def load_policy_corpus(
    data_dir: str | Path = POLICY_DATA_DIR,
) -> tuple[dict[str, Any], ...]:
    data_path = Path(data_dir)
    list_rows = load_json_rows(data_path / "service_list.json")
    detail_by_id = _by_service_id(
        load_json_rows(data_path / "service_detail.json"), source="service_detail.json"
    )
    conditions_by_id = _by_service_id(
        load_json_rows(data_path / "support_conditions.json"),
        source="support_conditions.json",
    )

    policies = []
    seen_ids: set[str] = set()
    for list_item in list_rows:
        service_id = normalize_text(list_item.get("서비스ID"))
        if not service_id or service_id in seen_ids:
            raise ValueError(
                f"service_list.json의 서비스ID가 없거나 중복입니다: {service_id}"
            )
        seen_ids.add(service_id)
        policies.append(
            normalize_policy_record(
                list_item,
                detail_by_id.get(service_id, {}),
                conditions_by_id.get(service_id, {}),
            )
        )

    if set(detail_by_id) != seen_ids or set(conditions_by_id) != seen_ids:
        raise ValueError("정책 원천 파일 사이의 서비스ID 집합이 일치하지 않습니다.")
    return tuple(policies)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_policy_search_document(policy: dict[str, Any]) -> str:
    fields = (
        ("정책명", "policy_name", 120),
        ("정책목적", "purpose", 220),
        ("서비스분야", "category", 80),
        ("지원유형", "support_type", 80),
        ("지원대상", "target_audience", 320),
        ("지원내용", "benefits", 320),
        ("선정기준", "selection_criteria", 220),
        ("신청기한", "application_period", 100),
        ("신청방법", "application_method", 100),
    )
    lines = []
    for label, key, limit in fields:
        value = normalize_text(policy.get(key))
        if value:
            lines.append(f"{label}: {_clip(value, limit)}")
    return "\n".join(lines)


def build_direct_policy_query(policy: dict[str, Any]) -> dict[str, str]:
    detail = policy.get("상세정보") if isinstance(policy.get("상세정보"), dict) else {}
    list_item = (
        policy.get("목록정보") if isinstance(policy.get("목록정보"), dict) else {}
    )
    return {
        "service_id": normalize_text(
            detail.get("서비스ID") or list_item.get("서비스ID")
        ),
        "policy_name": normalize_text(
            detail.get("서비스명") or list_item.get("서비스명")
        ),
        "purpose": normalize_text(
            detail.get("서비스목적") or list_item.get("서비스목적요약")
        ),
        "category": normalize_text(list_item.get("서비스분야")),
        "support_type": normalize_text(
            detail.get("지원유형") or list_item.get("지원유형")
        ),
        "target_audience": normalize_text(
            detail.get("지원대상") or list_item.get("지원대상")
        ),
        "selection_criteria": normalize_text(
            detail.get("선정기준") or detail.get("제외조건")
        ),
        "benefits": normalize_text(detail.get("지원내용") or list_item.get("지원내용")),
        "application_period": normalize_text(
            detail.get("신청기한") or list_item.get("신청기한")
        ),
        "application_method": normalize_text(
            detail.get("신청방법") or list_item.get("신청방법")
        ),
        "effective_date": normalize_date(detail.get("시행일")),
    }


def source_hashes(data_dir: str | Path = POLICY_DATA_DIR) -> dict[str, str]:
    data_path = Path(data_dir)
    result = {}
    for name in ("service_list.json", "service_detail.json", "support_conditions.json"):
        digest = hashlib.sha256()
        with (data_path / name).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result
