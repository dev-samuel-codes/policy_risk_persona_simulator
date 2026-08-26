import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from backend.ai_simulation_core.policies import gov24_openapi


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def resource_rows(path: str) -> list[dict]:
    rows = []
    for service_id in ("policy-1", "policy-2"):
        row = {"서비스ID": service_id, "서비스명": f"정책 {service_id[-1]}"}
        if path.endswith("serviceList"):
            row["등록일시"] = "20260815000000"
        rows.append(row)
    return rows


def fake_urlopen(request, *, timeout: float):
    del timeout
    parsed = urlparse(request.full_url)
    query = parse_qs(parsed.query)
    if query["serviceKey"] != ["test-key"]:
        raise AssertionError("인증키가 요청에 포함되지 않았습니다.")
    page = int(query["page"][0])
    rows = resource_rows(parsed.path)
    selected = rows[page - 1 : page]
    return FakeResponse(
        {
            "page": page,
            "perPage": 1,
            "currentCount": len(selected),
            "matchCount": len(rows),
            "totalCount": len(rows),
            "data": selected,
        }
    )


class Gov24OpenAPITest(unittest.TestCase):
    def test_fetches_all_three_resources_and_all_pages(self) -> None:
        client = gov24_openapi.Gov24OpenAPIClient(
            service_key="test-key",
            page_size=1,
            retries=0,
        )
        with patch.object(gov24_openapi, "urlopen", side_effect=fake_urlopen):
            snapshot = client.fetch_snapshot()

        self.assertEqual(
            set(snapshot),
            {"service_list.json", "service_detail.json", "support_conditions.json"},
        )
        self.assertEqual(snapshot["service_list.json"]["count"], 2)
        self.assertEqual(len(snapshot["service_list.json"]["pages"]), 2)
        self.assertEqual(
            snapshot["service_detail.json"]["source"]["api_type"],
            "OpenAPI",
        )
        self.assertEqual(
            {payload["fetched_at"] for payload in snapshot.values()},
            {snapshot["service_list.json"]["fetched_at"]},
        )

    def test_rejects_service_id_mismatch_before_writing(self) -> None:
        client = gov24_openapi.Gov24OpenAPIClient(
            service_key="test-key",
            page_size=1,
            retries=0,
        )
        with patch.object(gov24_openapi, "urlopen", side_effect=fake_urlopen):
            snapshot = client.fetch_snapshot()

        invalid = copy.deepcopy(snapshot)
        invalid["service_detail.json"]["data"][0]["서비스ID"] = "other-id"
        with self.assertRaisesRegex(
            gov24_openapi.Gov24OpenAPIError,
            "서비스ID 집합",
        ):
            gov24_openapi.validate_policy_snapshot(invalid)

    def test_directory_swap_keeps_previous_snapshot_as_backup(self) -> None:
        client = gov24_openapi.Gov24OpenAPIClient(
            service_key="test-key",
            page_size=1,
            retries=0,
        )
        with patch.object(gov24_openapi, "urlopen", side_effect=fake_urlopen):
            snapshot = client.fetch_snapshot()

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "policies"
            target.mkdir()
            for resource in gov24_openapi.RESOURCES:
                (target / resource.filename).write_text("old", encoding="utf-8")

            backup = gov24_openapi.write_policy_snapshot(snapshot, target_dir=target)

            self.assertIsNotNone(backup)
            self.assertEqual((backup / "service_list.json").read_text(), "old")
            updated = json.loads((target / "service_list.json").read_text())
            self.assertEqual(updated["count"], 2)

    def test_reads_key_from_env_file_without_persisting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "GOV24_OPENAPI_SERVICE_KEY='encoded%2Bkey'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                key = gov24_openapi.service_key_from_environment(env_file)

        self.assertEqual(key, "encoded+key")

    def test_http_error_does_not_expose_service_key(self) -> None:
        secret = "very-secret-service-key"
        client = gov24_openapi.Gov24OpenAPIClient(
            service_key=secret,
            page_size=1,
            retries=0,
        )

        def unauthorized(request, *, timeout: float):
            del timeout
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        with (
            patch.object(gov24_openapi, "urlopen", side_effect=unauthorized),
            self.assertRaises(gov24_openapi.Gov24OpenAPIError) as raised,
        ):
            client.fetch_resource(gov24_openapi.RESOURCES[0])

        self.assertNotIn(secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
