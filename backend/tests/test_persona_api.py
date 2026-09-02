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
                    "selection_cohort": "boundary",
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

    def test_region_boundary_candidate_query_keeps_selection_cohort(self) -> None:
        expected = [
            {
                "uuid": "other-region-1",
                "province": "부산",
                "district": "부산-해운대구",
                "age": 28,
                "match": {
                    "region_match": False,
                    "age_cohort": "eligible",
                    "age_match_reason": "within_range",
                    "selection_cohort": "region_boundary",
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
                    "province": "서울",
                    "age_min": 19,
                    "age_max": 34,
                    "cohort": "region_boundary",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cohort"], "region_boundary")
        self.assertEqual(response.json()["candidates"], expected)
        candidates.assert_called_once_with(
            region_scope="specific",
            province="서울",
            district="",
            age_min=19,
            age_max=34,
            cohort="region_boundary",
            limit=12,
            seed=0,
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

    def test_manual_simulation_accepts_declared_region_boundary(self) -> None:
        personas = [
            {
                "uuid": f"other-region-{index}",
                "occupation": "교사",
                "age": 28 + index,
                "province": "부산",
                "district": "부산-해운대구",
            }
            for index in range(3)
        ]
        payload = {
            "policy": {
                "policy_name": "서울 청년 지원",
                "benefits": "월 10만원 지원",
                "region_scope": "specific",
                "region_province": "서울",
                "region_district": "서울-서초구",
                "age_min": 19,
                "age_max": 34,
            },
            "selection_mode": "manual",
            "persona_ids": [persona["uuid"] for persona in personas],
            "selection_cohorts": ["region_boundary"] * 3,
        }
        with (
            patch.object(api, "resolve_personas", return_value=personas),
            patch.object(api, "search_similar_policies", return_value={"results": []}),
            patch.object(api, "save_active_policy"),
            patch.object(api, "write_simulation_job"),
            patch.object(api.simulation_executor, "submit"),
        ):
            response = self.client.post("/api/simulations", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selection_cohorts"], ["region_boundary"] * 3)
        self.assertTrue(
            all(
                match["selection_cohort"] == "region_boundary"
                and match["region_match"] is False
                for match in response.json()["selection_match"]
            )
        )

    def test_manual_simulation_infers_legacy_age_boundary_without_cohorts(self) -> None:
        personas = [
            {
                "uuid": "eligible",
                "occupation": "사무원",
                "age": 19,
                "province": "서울",
                "district": "서울-서초구",
            },
            {
                "uuid": "lower-boundary",
                "occupation": "학생",
                "age": 18,
                "province": "서울",
                "district": "서울-서초구",
            },
            {
                "uuid": "upper-boundary",
                "occupation": "연구원",
                "age": 35,
                "province": "서울",
                "district": "서울-서초구",
            },
        ]
        payload = {
            "policy": {
                "policy_name": "서울 청년 지원",
                "benefits": "월 10만원 지원",
                "region_scope": "specific",
                "region_province": "서울",
                "region_district": "서울-서초구",
                "age_min": 19,
                "age_max": 34,
            },
            "selection_mode": "manual",
            "persona_ids": [persona["uuid"] for persona in personas],
        }
        with (
            patch.object(api, "resolve_personas", return_value=personas),
            patch.object(api, "search_similar_policies", return_value={"results": []}),
            patch.object(api, "save_active_policy"),
            patch.object(api, "write_simulation_job"),
            patch.object(api.simulation_executor, "submit"),
        ):
            response = self.client.post("/api/simulations", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["selection_cohorts"],
            ["eligible", "boundary", "boundary"],
        )

    def test_manual_simulation_rejects_undeclared_region_boundary(self) -> None:
        personas = [
            {
                "uuid": f"other-region-{index}",
                "occupation": "교사",
                "age": 28 + index,
                "province": "부산",
                "district": "부산-해운대구",
            }
            for index in range(3)
        ]
        payload = {
            "policy": {
                "policy_name": "서울 청년 지원",
                "benefits": "월 10만원 지원",
                "region_scope": "specific",
                "region_province": "서울",
                "age_min": 19,
                "age_max": 34,
            },
            "selection_mode": "manual",
            "persona_ids": [persona["uuid"] for persona in personas],
        }
        with patch.object(api, "resolve_personas", return_value=personas):
            response = self.client.post("/api/simulations", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("적용 지역", response.json()["detail"])

    def test_manual_simulation_rejects_false_region_boundary_declaration(self) -> None:
        personas = [
            {
                "uuid": f"other-region-{index}",
                "occupation": "교사",
                "age": 28 + index,
                "province": "부산",
                "district": "부산-해운대구",
            }
            for index in range(3)
        ]
        payload = {
            "policy": {
                "policy_name": "서울 청년 지원",
                "benefits": "월 10만원 지원",
                "region_scope": "specific",
                "region_province": "서울",
                "age_min": 19,
                "age_max": 34,
            },
            "selection_mode": "manual",
            "persona_ids": [persona["uuid"] for persona in personas],
            "selection_cohorts": ["eligible"] * 3,
        }
        with patch.object(api, "resolve_personas", return_value=personas):
            response = self.client.post("/api/simulations", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("후보 유형", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
