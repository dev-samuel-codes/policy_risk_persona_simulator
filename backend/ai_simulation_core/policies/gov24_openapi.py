"""정부24 공공서비스(보조금24) OpenAPI 수집기."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen

from backend.ai_simulation_core.policies.policy_corpus import POLICY_DATA_DIR


DEFAULT_BASE_URL = "https://api.odcloud.kr/api"
SERVICE_KEY_ENV_NAMES = (
    "GOV24_OPENAPI_SERVICE_KEY",
    "DATA_GO_KR_SERVICE_KEY",
    "ODCLOUD_SERVICE_KEY",
)
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Gov24Resource:
    name: str
    path: str
    filename: str
    required_fields: tuple[str, ...]


RESOURCES = (
    Gov24Resource(
        name="service_list",
        path="/gov24/v3/serviceList",
        filename="service_list.json",
        required_fields=("서비스ID", "서비스명", "등록일시"),
    ),
    Gov24Resource(
        name="service_detail",
        path="/gov24/v3/serviceDetail",
        filename="service_detail.json",
        required_fields=("서비스ID", "서비스명"),
    ),
    Gov24Resource(
        name="support_conditions",
        path="/gov24/v3/supportConditions",
        filename="support_conditions.json",
        required_fields=("서비스ID", "서비스명"),
    ),
)


class Gov24OpenAPIError(RuntimeError):
    """정부24 OpenAPI 요청 또는 스냅샷 검증 실패."""


def service_key_from_environment(env_file: str | Path | None = None) -> str:
    for name in SERVICE_KEY_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            # 공공데이터포털이 발급한 인코딩 키와 디코딩 키를 모두 허용한다.
            return unquote(value)

    if env_file is not None and Path(env_file).is_file():
        for raw_line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in SERVICE_KEY_ENV_NAMES and value.strip():
                return unquote(value.strip().strip("'\""))

    names = ", ".join(SERVICE_KEY_ENV_NAMES)
    raise Gov24OpenAPIError(
        f"정부24 OpenAPI 인증키가 없습니다. 다음 환경 변수 중 하나를 설정하세요: {names}"
    )


def _positive_int(value: object, *, field: str, resource: str, page: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise Gov24OpenAPIError(
            f"{resource} {page}페이지 응답의 {field} 값이 정수가 아닙니다."
        ) from error
    if normalized < 0:
        raise Gov24OpenAPIError(
            f"{resource} {page}페이지 응답의 {field} 값이 음수입니다."
        )
    return normalized


@dataclass(frozen=True)
class Gov24OpenAPIClient:
    service_key: str
    base_url: str = DEFAULT_BASE_URL
    page_size: int = 1000
    timeout: float = 30.0
    retries: int = 3

    def __post_init__(self) -> None:
        if not self.service_key.strip():
            raise ValueError("정부24 OpenAPI 인증키가 비어 있습니다.")
        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError("page_size는 1에서 1000 사이여야 합니다.")
        if self.timeout <= 0:
            raise ValueError("timeout은 0보다 커야 합니다.")
        if self.retries < 0:
            raise ValueError("retries는 0 이상이어야 합니다.")

    def fetch_snapshot(self) -> dict[str, dict[str, Any]]:
        fetched_at = datetime.now(timezone.utc).isoformat()
        snapshot = {
            resource.filename: self.fetch_resource(
                resource,
                fetched_at=fetched_at,
            )
            for resource in RESOURCES
        }
        validate_policy_snapshot(snapshot)
        return snapshot

    def fetch_resource(
        self,
        resource: Gov24Resource,
        *,
        fetched_at: str | None = None,
    ) -> dict[str, Any]:
        first_page = self._fetch_page(resource, page=1)
        total_count = first_page["total_count"]
        if total_count < 1:
            raise Gov24OpenAPIError(f"{resource.name} 응답에 정책이 없습니다.")

        page_count = (total_count + self.page_size - 1) // self.page_size
        pages = [first_page]
        for page_number in range(2, page_count + 1):
            pages.append(self._fetch_page(resource, page=page_number))

        rows = [row for page in pages for row in page["data"]]
        if len(rows) != total_count:
            raise Gov24OpenAPIError(
                f"{resource.name} 전체 건수 불일치: API {total_count:,}건, "
                f"수집 {len(rows):,}건"
            )
        _validate_resource_rows(resource, rows)

        return {
            "resource": resource.name,
            "source": {
                "provider": "행정안전부",
                "dataset": "대한민국 공공서비스(혜택) 정보",
                "api_type": "OpenAPI",
                "base_url": self.base_url.rstrip("/"),
                "path": resource.path,
            },
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
            "page_size": self.page_size,
            "pages": [
                {
                    "page": page["page"],
                    "currentCount": page["current_count"],
                    "matchCount": page["match_count"],
                    "totalCount": page["total_count"],
                }
                for page in pages
            ],
            "count": len(rows),
            "data": rows,
        }

    def _fetch_page(self, resource: Gov24Resource, *, page: int) -> dict[str, Any]:
        params = {
            "page": page,
            "perPage": self.page_size,
            "returnType": "JSON",
            "serviceKey": unquote(self.service_key.strip()),
        }
        request = Request(
            f"{self.base_url.rstrip('/')}{resource.path}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "CivicEcho/0.1",
            },
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8-sig")
                payload = json.loads(body)
                return self._normalize_page(
                    payload,
                    resource=resource,
                    requested_page=page,
                )
            except HTTPError as error:
                last_error = error
                if error.code not in RETRYABLE_HTTP_STATUS or attempt >= self.retries:
                    raise Gov24OpenAPIError(
                        f"{resource.name} {page}페이지 HTTP 오류({error.code})"
                    ) from error
            except (URLError, TimeoutError) as error:
                last_error = error
                if attempt >= self.retries:
                    raise Gov24OpenAPIError(
                        f"{resource.name} {page}페이지 연결 실패"
                    ) from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise Gov24OpenAPIError(
                    f"{resource.name} {page}페이지 JSON 파싱 실패"
                ) from error

            time.sleep(min(2**attempt, 8))

        raise Gov24OpenAPIError(
            f"{resource.name} {page}페이지 요청 실패"
        ) from last_error

    @staticmethod
    def _normalize_page(
        payload: object,
        *,
        resource: Gov24Resource,
        requested_page: int,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise Gov24OpenAPIError(
                f"{resource.name} {requested_page}페이지 응답이 JSON 객체가 아닙니다."
            )
        if payload.get("code") not in (None, 0, "0"):
            message = str(payload.get("msg") or "알 수 없는 API 오류").strip()
            raise Gov24OpenAPIError(
                f"{resource.name} {requested_page}페이지 API 오류: {message}"
            )

        rows = payload.get("data")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise Gov24OpenAPIError(
                f"{resource.name} {requested_page}페이지 data 형식이 올바르지 않습니다."
            )

        page = _positive_int(
            payload.get("page", requested_page),
            field="page",
            resource=resource.name,
            page=requested_page,
        )
        if page != requested_page:
            raise Gov24OpenAPIError(
                f"{resource.name} 요청 페이지({requested_page})와 응답 페이지({page})가 다릅니다."
            )
        current_count = _positive_int(
            payload.get("currentCount", len(rows)),
            field="currentCount",
            resource=resource.name,
            page=page,
        )
        if current_count != len(rows):
            raise Gov24OpenAPIError(
                f"{resource.name} {page}페이지 currentCount와 data 건수가 다릅니다."
            )

        return {
            "page": page,
            "current_count": current_count,
            "match_count": _positive_int(
                payload.get("matchCount", payload.get("totalCount")),
                field="matchCount",
                resource=resource.name,
                page=page,
            ),
            "total_count": _positive_int(
                payload.get("totalCount"),
                field="totalCount",
                resource=resource.name,
                page=page,
            ),
            "data": rows,
        }


def _validate_resource_rows(resource: Gov24Resource, rows: list[dict[str, Any]]) -> None:
    service_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing = [field for field in resource.required_fields if not row.get(field)]
        if missing:
            raise Gov24OpenAPIError(
                f"{resource.name} {index}번째 행에 필수 필드가 없습니다: "
                + ", ".join(missing)
            )
        service_ids.append(str(row["서비스ID"]).strip())
    if len(service_ids) != len(set(service_ids)):
        raise Gov24OpenAPIError(f"{resource.name}에 중복 서비스ID가 있습니다.")


def validate_policy_snapshot(snapshot: dict[str, dict[str, Any]]) -> None:
    expected_files = {resource.filename for resource in RESOURCES}
    if set(snapshot) != expected_files:
        raise Gov24OpenAPIError("정책 스냅샷에 필요한 세 OpenAPI 응답이 모두 없습니다.")

    id_sets: dict[str, set[str]] = {}
    fetched_at_values: set[str] = set()
    for resource in RESOURCES:
        payload = snapshot[resource.filename]
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise Gov24OpenAPIError(f"{resource.filename}의 data 형식이 올바르지 않습니다.")
        _validate_resource_rows(resource, rows)
        if payload.get("count") != len(rows):
            raise Gov24OpenAPIError(f"{resource.filename}의 count와 data 건수가 다릅니다.")
        id_sets[resource.name] = {str(row["서비스ID"]).strip() for row in rows}
        fetched_at_values.add(str(payload.get("fetched_at") or "").strip())

    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        counts = ", ".join(f"{name}={len(ids):,}" for name, ids in id_sets.items())
        raise Gov24OpenAPIError(f"OpenAPI별 서비스ID 집합이 일치하지 않습니다: {counts}")
    if "" in fetched_at_values or len(fetched_at_values) != 1:
        raise Gov24OpenAPIError("세 OpenAPI 응답의 수집 시점이 일치하지 않습니다.")


def write_policy_snapshot(
    snapshot: dict[str, dict[str, Any]],
    *,
    target_dir: str | Path = POLICY_DATA_DIR,
) -> Path | None:
    """검증된 세 응답을 디렉터리 단위로 교체하고 이전 스냅샷을 보존한다."""

    validate_policy_snapshot(snapshot)
    target = Path(target_dir)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    expected_files = {resource.filename for resource in RESOURCES}
    if target.exists():
        unexpected = {
            path.name for path in target.iterdir() if path.name not in expected_files
        }
        if unexpected:
            raise Gov24OpenAPIError(
                "정책 데이터 디렉터리에 보존해야 할 다른 파일이 있습니다: "
                + ", ".join(sorted(unexpected))
            )

    staging = Path(mkdtemp(prefix=".gov24-staging-", dir=parent))
    try:
        for filename, payload in snapshot.items():
            (staging / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = parent / f".gov24-backup-{timestamp}"
        if target.exists():
            if backup.exists():
                raise Gov24OpenAPIError(f"백업 경로가 이미 존재합니다: {backup}")
            target.rename(backup)
        else:
            backup = None

        try:
            staging.rename(target)
        except Exception:
            if backup is not None and not target.exists():
                backup.rename(target)
            raise
        return backup
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise


def fetch_and_write_policy_snapshot(
    *,
    target_dir: str | Path = POLICY_DATA_DIR,
    service_key: str | None = None,
    page_size: int = 1000,
    timeout: float = 30.0,
    retries: int = 3,
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    client = Gov24OpenAPIClient(
        service_key=service_key or service_key_from_environment(),
        page_size=page_size,
        timeout=timeout,
        retries=retries,
    )
    snapshot = client.fetch_snapshot()
    backup = write_policy_snapshot(snapshot, target_dir=target_dir)
    return snapshot, backup
