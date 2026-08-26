import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import api


class PersonaApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_options_are_cacheable_and_keep_original_district_values(self) -> None:
        options = {
            "provinces": [
                {
                    "province": "서울",
                    "districts": ["서울-서초구", "서울-강남구"],
                }
            ]
        }
        with patch.object(api, "get_region_options", return_value=options):
            response = self.client.get("/api/personas/options")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), options)
        self.assertEqual(response.headers["cache-control"], "public, max-age=3600")
        self.assertTrue(response.headers["etag"].startswith('"'))

    def test_candidate_query_passes_structured_filters(self) -> None:
        expected = [
            {
                "uuid": "boundary-1",
                "province": "서울",
                "district": "서울-서초구",
                "age": 35,
                "match": {
                    "region_match": True,
                    "age_cohort": "boundary",
                    "age_match_reason": "above_maximum",
                },
            }
        ]
        with patch.object(
            api,
            "get_persona_candidates",
            return_value=expected,
        ) as candidates:
            response = self.client.get(
                "/api/personas/candidates",
                params={
                    "region_scope": "specific",
                    "province": " 서울 ",
                    "district": " 서울-서초구 ",
                    "age_min": 19,
                    "age_max": 34,
                    "cohort": "boundary",
                    "limit": 12,
                    "seed": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["candidates"], expected)
        candidates.assert_called_once_with(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="boundary",
            limit=12,
            seed=4,
        )

    def test_candidate_query_reports_validation_error_without_fallback(self) -> None:
        with patch.object(
            api,
            "get_persona_candidates",
            side_effect=ValueError("특정 지역 조회에는 province가 필요합니다."),
        ):
            response = self.client.get(
                "/api/personas/candidates",
                params={"region_scope": "specific"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("province", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
