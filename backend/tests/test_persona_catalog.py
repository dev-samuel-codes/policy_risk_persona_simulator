import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from backend.ai_simulation_core.personas import persona_catalog


class PersonaCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        rows = [
            {
                "uuid": "seoul-eligible-low",
                "occupation": "사무원",
                "sex": "남성",
                "age": 19,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "서울에서 일하는 청년",
                "professional_persona": "신입 사무원",
                "family_persona": "1인 가구",
            },
            {
                "uuid": "seoul-eligible-high",
                "occupation": "연구원",
                "sex": "여성",
                "age": 34,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "서울 거주 연구원",
                "professional_persona": "연구원",
                "family_persona": "2인 가구",
            },
            {
                "uuid": "seoul-boundary-low",
                "occupation": "학생",
                "sex": "여성",
                "age": 18,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "진학을 준비하는 학생",
                "professional_persona": "학생",
                "family_persona": "부모 동거",
            },
            {
                "uuid": "seoul-boundary-high",
                "occupation": "디자이너",
                "sex": "남성",
                "age": 35,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "주거비를 부담하는 디자이너",
                "professional_persona": "디자이너",
                "family_persona": "1인 가구",
            },
            {
                "uuid": "seoul-too-old",
                "occupation": "회계사",
                "sex": "여성",
                "age": 52,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "중년 회계사",
                "professional_persona": "회계사",
                "family_persona": "4인 가구",
            },
            {
                "uuid": "busan-eligible",
                "occupation": "교사",
                "sex": "여성",
                "age": 28,
                "province": "부산",
                "district": "부산-해운대구",
                "persona": "부산 거주 교사",
                "professional_persona": "교사",
                "family_persona": "2인 가구",
            },
            {
                "uuid": "busan-boundary-low",
                "occupation": "학생",
                "sex": "여성",
                "age": 18,
                "province": "부산",
                "district": "부산-해운대구",
                "persona": "부산 거주 학생",
                "professional_persona": "학생",
                "family_persona": "부모 동거",
            },
            {
                "uuid": "busan-boundary-high",
                "occupation": "디자이너",
                "sex": "남성",
                "age": 35,
                "province": "부산",
                "district": "부산-해운대구",
                "persona": "부산 거주 디자이너",
                "professional_persona": "디자이너",
                "family_persona": "1인 가구",
            },
            {
                "uuid": "busan-outside-age",
                "occupation": "회계사",
                "sex": "여성",
                "age": 52,
                "province": "부산",
                "district": "부산-해운대구",
                "persona": "부산 거주 회계사",
                "professional_persona": "회계사",
                "family_persona": "4인 가구",
            },
            {
                "uuid": "seoul-official",
                "occupation": "지방 공무원",
                "sex": "남성",
                "age": 30,
                "province": "서울",
                "district": "서울-서초구",
                "persona": "지방 공무원",
                "professional_persona": "공무원",
                "family_persona": "3인 가구",
            },
        ]
        midpoint = 4
        self.files = [directory / "0000.parquet", directory / "0001.parquet"]
        pq.write_table(
            pa.Table.from_pylist(rows[:midpoint]), self.files[0], row_group_size=2
        )
        pq.write_table(
            pa.Table.from_pylist(rows[midpoint:]), self.files[1], row_group_size=2
        )
        self.files_patch = patch.object(
            persona_catalog,
            "get_local_parquet_files",
            return_value=self.files,
        )
        self.files_patch.start()
        persona_catalog.clear_persona_catalog_caches()

    def tearDown(self) -> None:
        self.files_patch.stop()
        persona_catalog.clear_persona_catalog_caches()
        self.temporary_directory.cleanup()

    def test_region_options_use_actual_province_and_district_values(self) -> None:
        options = persona_catalog.get_region_options(auto_download=False)

        self.assertEqual(
            options,
            {
                "provinces": [
                    {"province": "부산", "districts": ["부산-해운대구"]},
                    {"province": "서울", "districts": ["서울-서초구"]},
                ]
            },
        )

    def test_region_matches_normalizes_only_verified_province_prefixes(self) -> None:
        persona = {
            "province": "서울",
            "district": "서울-서초구",
        }

        self.assertTrue(
            persona_catalog.region_matches(
                persona,
                region_scope="specific",
                province="서울",
                district="서초구",
            )
        )
        self.assertTrue(
            persona_catalog.region_matches(
                {**persona, "district": "서초구"},
                region_scope="specific",
                province="서울",
                district="서울-서초구",
            )
        )
        self.assertFalse(
            persona_catalog.region_matches(
                persona,
                region_scope="specific",
                province="서울",
                district="강남구",
            )
        )
        self.assertFalse(
            persona_catalog.region_matches(
                {"province": "서울", "district": "서울-강서구"},
                region_scope="specific",
                province="서울",
                district="서구",
            )
        )
        self.assertFalse(
            persona_catalog.region_matches(
                {"province": "부산", "district": "부산-중구"},
                region_scope="specific",
                province="서울",
                district="서울-중구",
            )
        )
        self.assertFalse(
            persona_catalog.region_matches(
                {"province": "서울", "district": ""},
                region_scope="specific",
                province="서울",
                district="서초구",
            )
        )
        self.assertTrue(
            persona_catalog.region_matches(
                persona,
                region_scope="specific",
                province="서울",
            )
        )
        for province, short_district in (
            ("부산", "부산진구"),
            ("세종", "세종시"),
            ("제주", "제주시"),
        ):
            with self.subTest(province=province, district=short_district):
                self.assertTrue(
                    persona_catalog.region_matches(
                        {
                            "province": province,
                            "district": f"{province}-{short_district}",
                        },
                        region_scope="specific",
                        province=province,
                        district=short_district,
                    )
                )

    def test_eligible_candidates_apply_region_and_age_hard_filters(self) -> None:
        candidates = persona_catalog.get_persona_candidates(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="eligible",
            limit=12,
            seed=7,
            auto_download=False,
        )

        self.assertEqual(
            {candidate["uuid"] for candidate in candidates},
            {"seoul-eligible-low", "seoul-eligible-high"},
        )
        self.assertTrue(
            all(candidate["match"]["region_match"] for candidate in candidates)
        )
        self.assertTrue(
            all(
                candidate["match"]["age_cohort"] == "eligible"
                for candidate in candidates
            )
        )
        self.assertTrue(
            all(
                candidate["match"]["selection_cohort"] == "eligible"
                for candidate in candidates
            )
        )
        self.assertTrue(all("persona" in candidate for candidate in candidates))

    def test_region_boundary_is_other_region_and_age_eligible_only(self) -> None:
        candidates = persona_catalog.get_persona_candidates(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="region_boundary",
            limit=12,
            seed=9,
            auto_download=False,
        )

        self.assertEqual(
            {candidate["uuid"] for candidate in candidates},
            {"busan-eligible"},
        )
        self.assertTrue(
            all(candidate["match"]["region_match"] is False for candidate in candidates)
        )
        self.assertTrue(
            all(
                candidate["match"]["age_cohort"] == "eligible"
                and candidate["match"]["selection_cohort"] == "region_boundary"
                for candidate in candidates
            )
        )

    def test_boundary_is_only_one_year_outside_and_never_falls_back(self) -> None:
        candidates = persona_catalog.get_persona_candidates(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="boundary",
            limit=12,
            seed=3,
            auto_download=False,
        )

        self.assertEqual(
            {candidate["uuid"] for candidate in candidates},
            {"seoul-boundary-low", "seoul-boundary-high"},
        )
        self.assertNotIn("seoul-too-old", {item["uuid"] for item in candidates})
        self.assertNotIn("busan-eligible", {item["uuid"] for item in candidates})
        self.assertTrue(
            all(
                candidate["match"]["selection_cohort"] == "boundary"
                for candidate in candidates
            )
        )

        no_match = persona_catalog.get_persona_candidates(
            region_scope="specific",
            province="서울",
            district="서울-강남구",
            age_min=19,
            age_max=34,
            cohort="eligible",
            limit=12,
            auto_download=False,
        )
        self.assertEqual(no_match, [])

    def test_boundary_without_age_condition_is_empty(self) -> None:
        self.assertEqual(
            persona_catalog.get_persona_candidates(
                cohort="boundary",
                auto_download=False,
            ),
            [],
        )

    def test_region_boundary_rejects_nationwide_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "특정 지역"):
            persona_catalog.get_persona_candidates(
                region_scope="nationwide",
                cohort="region_boundary",
                auto_download=False,
            )

    def test_nationwide_filter_rejects_specific_region_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "특정 지역 조회"):
            persona_catalog.get_persona_candidates(
                region_scope="nationwide",
                province="서울",
                auto_download=False,
            )

        with self.assertRaisesRegex(ValueError, "특정 지역 조회"):
            persona_catalog.get_persona_candidates(
                region_scope="nationwide",
                district="서울-서초구",
                auto_download=False,
            )

    def test_resolve_personas_preserves_requested_order(self) -> None:
        resolved = persona_catalog.resolve_personas(
            ["seoul-boundary-high", "seoul-eligible-low"],
            auto_download=False,
        )

        self.assertEqual(
            [persona["uuid"] for persona in resolved],
            ["seoul-boundary-high", "seoul-eligible-low"],
        )
        self.assertEqual(resolved[0]["professional_persona"], "디자이너")

        with self.assertRaisesRegex(ValueError, "존재하지 않는"):
            persona_catalog.resolve_personas(
                ["missing-persona"],
                auto_download=False,
            )

    def test_selection_rejects_region_age_and_public_servant_mismatches(self) -> None:
        common = {
            "region_scope": "specific",
            "province": "서울",
            "district": "서울-서초구",
            "age_min": 19,
            "age_max": 34,
        }
        with self.assertRaisesRegex(ValueError, "적용 지역"):
            persona_catalog.validate_persona_selection(
                [
                    {
                        "uuid": "other-region",
                        "occupation": "교사",
                        "age": 28,
                        "province": "부산",
                        "district": "부산-해운대구",
                    }
                ],
                **common,
            )
        with self.assertRaisesRegex(ValueError, "연령 경계선"):
            persona_catalog.validate_persona_selection(
                [
                    {
                        "uuid": "too-old",
                        "occupation": "회계사",
                        "age": 52,
                        "province": "서울",
                        "district": "서울-서초구",
                    }
                ],
                **common,
            )
        with self.assertRaisesRegex(ValueError, "공무원"):
            persona_catalog.validate_persona_selection(
                [
                    {
                        "uuid": "official",
                        "occupation": "지방 공무원",
                        "age": 30,
                        "province": "서울",
                        "district": "서울-서초구",
                    }
                ],
                **common,
            )

    def test_selection_allows_region_boundary_only_when_explicitly_enabled(self) -> None:
        common = {
            "region_scope": "specific",
            "province": "서울",
            "district": "서울-서초구",
            "age_min": 19,
            "age_max": 34,
        }
        evidence = persona_catalog.validate_persona_selection(
            [
                {
                    "uuid": "other-region-eligible",
                    "occupation": "교사",
                    "age": 28,
                    "province": "부산",
                    "district": "부산-해운대구",
                }
            ],
            selection_cohorts=["region_boundary"],
            **common,
        )

        self.assertEqual(evidence[0]["selection_cohort"], "region_boundary")
        self.assertFalse(evidence[0]["region_match"])
        self.assertEqual(evidence[0]["age_cohort"], "eligible")

        with self.assertRaisesRegex(ValueError, "후보 유형"):
            persona_catalog.validate_persona_selection(
                [
                    {
                        "uuid": "other-region-boundary",
                        "occupation": "디자이너",
                        "age": 35,
                        "province": "부산",
                        "district": "부산-해운대구",
                    }
                ],
                selection_cohorts=["region_boundary"],
                **common,
            )

        with self.assertRaisesRegex(ValueError, "후보 유형"):
            persona_catalog.validate_persona_selection(
                [
                    {
                        "uuid": "other-region-eligible",
                        "occupation": "교사",
                        "age": 28,
                        "province": "부산",
                        "district": "부산-해운대구",
                    }
                ],
                selection_cohorts=["eligible"],
                **common,
            )


if __name__ == "__main__":
    unittest.main()
