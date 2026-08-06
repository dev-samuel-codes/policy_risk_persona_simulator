"""국가법령정보센터 공동활용 API 클라이언트."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_LAW_API_URL = "https://www.law.go.kr/DRF/lawService.do"
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class LawApiError(RuntimeError):
    """국가법령정보센터 API 요청 또는 응답 처리 실패."""


@dataclass(frozen=True)
class LawApiClient:
    """현행법령 본문을 JSON으로 조회한다."""

    oc: str
    timeout: float = 30.0
    retries: int = 3
    endpoint: str = DEFAULT_LAW_API_URL

    def __post_init__(self) -> None:
        if not self.oc.strip():
            raise ValueError("국가법령정보센터 API 인증값(OC)이 비어 있습니다.")
        if self.timeout <= 0:
            raise ValueError("timeout은 0보다 커야 합니다.")
        if self.retries < 0:
            raise ValueError("retries는 0 이상이어야 합니다.")

    def fetch_law(
        self,
        *,
        law_id: str | None = None,
        mst: str | None = None,
        article: str | None = None,
    ) -> dict[str, Any]:
        """법령 전체 또는 특정 조문을 조회한다.

        ``article``을 생략하면 법령 전체 조문을 반환한다. 법령 식별자는
        ``law_id`` 또는 ``mst`` 중 적어도 하나를 전달해야 한다.
        """

        normalized_law_id = _digits_or_none(law_id, field="law_id")
        normalized_mst = _digits_or_none(mst, field="mst")
        normalized_article = _digits_or_none(article, field="article")
        if normalized_law_id is None and normalized_mst is None:
            raise ValueError("law_id 또는 mst 중 하나가 필요합니다.")

        params = {
            "OC": self.oc.strip(),
            "target": "law",
            "type": "JSON",
        }
        if normalized_law_id is not None:
            params["ID"] = normalized_law_id
        else:
            params["MST"] = normalized_mst or ""
        if normalized_article is not None:
            params["JO"] = normalized_article

        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "policy-risk-persona-simulator/0.1",
            },
        )
        return self._request_json(request, law_id=normalized_law_id, mst=normalized_mst)

    def _request_json(
        self,
        request: Request,
        *,
        law_id: str | None,
        mst: str | None,
    ) -> dict[str, Any]:
        identifier = f"ID={law_id}" if law_id is not None else f"MST={mst}"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8-sig")
                    content_type = response.headers.get("Content-Type", "")
                if "html" in content_type.casefold() or body.lstrip().startswith("<"):
                    message = _html_error_message(body)
                    raise LawApiError(
                        f"법령 API HTML 오류 응답: {message} ({identifier})"
                    )
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise LawApiError(
                        f"법령 API 응답이 JSON 객체가 아닙니다: {identifier}"
                    )
                _raise_for_api_error(payload, identifier=identifier)
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_HTTP_STATUS or attempt >= self.retries:
                    raise LawApiError(
                        f"법령 API HTTP 오류({exc.code}): {identifier}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise LawApiError(f"법령 API 연결 실패: {identifier}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LawApiError(
                    f"법령 API JSON 응답 파싱 실패: {identifier}"
                ) from exc

            time.sleep(min(2**attempt, 8))

        raise LawApiError(f"법령 API 요청 실패: {identifier}") from last_error


def _digits_or_none(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if not normalized.isdigit():
        raise ValueError(f"{field}는 숫자로만 구성되어야 합니다: {normalized!r}")
    return normalized


def _raise_for_api_error(payload: dict[str, Any], *, identifier: str) -> None:
    """API가 HTTP 200과 함께 반환한 오류 메시지를 탐지한다."""

    candidates: list[dict[str, Any]] = [payload]
    candidates.extend(value for value in payload.values() if isinstance(value, dict))
    for candidate in candidates:
        result_code = candidate.get("resultCode") or candidate.get("결과코드")
        if result_code in (None, "", "00", "0", 0):
            continue
        message = (
            candidate.get("resultMsg")
            or candidate.get("resultMessage")
            or candidate.get("결과메시지")
            or "알 수 없는 API 오류"
        )
        raise LawApiError(f"법령 API 오류({result_code}): {message} ({identifier})")


def _html_error_message(body: str) -> str:
    without_scripts = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = unescape(re.sub(r"<[^>]+>", "\n", without_scripts))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preferred = [
        line
        for line in lines
        if "미신청" in line or "신청" in line or "접근" in line or "오류" in line
    ]
    summary = " / ".join(preferred[:2] or lines[:2])
    return summary or "내용을 확인할 수 없는 HTML 응답"
