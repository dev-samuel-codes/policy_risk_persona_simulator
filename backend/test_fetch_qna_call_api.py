from __future__ import annotations

import httpx2
import pytest

from backend.fetch_qna_call_api import (
    ApiResultError,
    build_pending_pages,
    merge_records,
    parse_page_payload,
    record_id,
    request_error_summary,
)


def sample_case(question: str = "지원 대상은 누구인가요?") -> dict[str, str]:
    return {
        "consQustCntn": question,
        "nobdAnsrCntn": "지원 대상 안내입니다.",
        "inttCode": "1140100",
        "kywdNm": "지원 대상",
        "datSourceNm": "국민콜110",
    }


def test_record_id_is_stable_for_same_case() -> None:
    case = sample_case()
    reordered = dict(reversed(list(case.items())))

    assert record_id(case) == record_id(reordered)


def test_merge_records_skips_existing_and_page_duplicates() -> None:
    existing = [{**sample_case(), "recordId": record_id(sample_case())}]
    new_case = sample_case("신청 방법은 무엇인가요?")

    merged, added, skipped = merge_records(
        existing,
        [sample_case(), new_case, new_case],
    )

    assert len(merged) == 2
    assert added == 1
    assert skipped == 2


def test_failed_pages_are_retried_before_unfinished_pages() -> None:
    assert build_pending_pages(
        total_pages=5,
        completed_pages={1, 2},
        failed_pages={4},
        start_page=1,
    ) == [4, 3, 5]


def test_parse_page_payload_accepts_live_response_shape() -> None:
    payload = {
        "header": {"resultCode": "00", "resultMsg": "정상"},
        "body": {
            "numOfRows": 100,
            "pageNo": 1,
            "totalCount": 1,
            "dataType": "JSON",
            "items": [sample_case()],
        },
    }

    page = parse_page_payload(payload)

    assert page.total_count == 1
    assert page.page_no == 1
    assert page.items == [sample_case()]


def test_request_error_summary_does_not_expose_service_key() -> None:
    request = httpx2.Request(
        "GET", "https://example.test/api?serviceKey=top-secret"
    )
    response = httpx2.Response(401, request=request)
    error = httpx2.HTTPStatusError(
        "unauthorized", request=request, response=response
    )

    summary = request_error_summary(error)

    assert summary == "HTTPStatusError: HTTP 401"
    assert "top-secret" not in summary


def test_parse_page_payload_wraps_invalid_counts_as_api_error() -> None:
    payload = {
        "header": {"resultCode": "00", "resultMsg": "정상"},
        "body": {"pageNo": "invalid", "totalCount": 1, "items": []},
    }

    with pytest.raises(ApiResultError, match="INVALID_NUMBER"):
        parse_page_payload(payload)
