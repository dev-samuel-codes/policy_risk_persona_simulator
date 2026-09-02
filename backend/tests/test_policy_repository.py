import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.ai_simulation_core.policies import policy_repository


class DirectPolicyRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = {
            "policy_name": "청년 주거 지원",
            "target_audience": "만 19세 이상 34세 이하 청년",
            "selection_criteria": "중위소득과 주거 여건을 심사",
            "application_period": "2026.08.01 ~ 2026.08.31",
            "effective_date": "2026-09-01",
            "required_documents": "신청서, 주민등록등본",
            "application_method": "온라인 신청",
            "contact": "정책지원과 02-0000-0000",
            "benefits": "월 20만 원 지원",
            "exclusion_conditions": "기존 수혜자 제외",
        }

    def test_build_direct_policy_matches_pipeline_schema(self) -> None:
        policy = policy_repository.build_direct_policy(self.fields)

        self.assertEqual(policy["입력출처"], "직접입력")
        self.assertEqual(policy["상세정보"]["서비스명"], "청년 주거 지원")
        self.assertEqual(policy["상세정보"]["지원내용"], "월 20만 원 지원")
        self.assertEqual(
            policy["상세정보"]["선정기준"],
            "중위소득과 주거 여건을 심사",
        )
        self.assertEqual(policy["상세정보"]["제외조건"], "기존 수혜자 제외")

    def test_missing_selection_criteria_is_not_filled_from_exclusion(self) -> None:
        fields = {**self.fields}
        fields.pop("selection_criteria")

        policy = policy_repository.build_direct_policy(fields)

        self.assertEqual(policy["상세정보"]["선정기준"], "")
        self.assertEqual(policy["상세정보"]["제외조건"], "기존 수혜자 제외")

    def test_policy_name_and_benefits_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "정책명"):
            policy_repository.build_direct_policy({"benefits": "혜택"})
        with self.assertRaisesRegex(ValueError, "혜택"):
            policy_repository.build_direct_policy({"policy_name": "정책"})

    def test_saved_direct_policy_becomes_active_policy(self) -> None:
        policy = policy_repository.build_direct_policy(self.fields)

        with tempfile.TemporaryDirectory() as directory:
            active_path = Path(directory) / "active_policy.json"
            with patch.object(policy_repository, "ACTIVE_POLICY_PATH", active_path):
                policy_repository.save_active_policy(policy)
                loaded = policy_repository.load_active_policy()
                selected = policy_repository.get_active_or_random_policy()

        self.assertEqual(loaded, policy)
        self.assertEqual(selected["상세정보"]["서비스명"], "청년 주거 지원")

    def test_direct_policy_keeps_structured_region_and_age_conditions(self) -> None:
        policy = policy_repository.build_direct_policy(
            {
                **self.fields,
                "region_scope": "specific",
                "region_province": "서울",
                "region_district": "서울-서초구",
                "age_min": 19,
                "age_max": 34,
                "age_basis": "dataset_age",
            }
        )

        self.assertEqual(policy["region_scope"], "specific")
        self.assertEqual(policy["region_province"], "서울")
        self.assertEqual(policy["region_district"], "서울-서초구")
        self.assertEqual(policy["age_min"], 19)
        self.assertEqual(policy["age_max"], 34)
        self.assertEqual(policy["age_basis"], "dataset_age")

    def test_invalid_structured_conditions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "전국 정책"):
            policy_repository.build_direct_policy(
                {
                    **self.fields,
                    "region_scope": "nationwide",
                    "region_province": "서울",
                }
            )
        with self.assertRaisesRegex(ValueError, "region_province"):
            policy_repository.build_direct_policy(
                {
                    **self.fields,
                    "region_scope": "specific",
                }
            )
        with self.assertRaisesRegex(ValueError, "age_min"):
            policy_repository.build_direct_policy(
                {
                    **self.fields,
                    "age_min": 35,
                    "age_max": 34,
                }
            )


if __name__ == "__main__":
    unittest.main()
