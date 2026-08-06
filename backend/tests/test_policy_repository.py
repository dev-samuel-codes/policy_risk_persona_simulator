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
        self.assertEqual(policy["상세정보"]["선정기준"], "기존 수혜자 제외")

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


if __name__ == "__main__":
    unittest.main()
