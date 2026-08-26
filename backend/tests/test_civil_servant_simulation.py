import unittest

from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    QUALITY_MODE,
    build_grounded_response,
    run_civil_servant_simulation,
    validate_civil_servant_response,
)


class CivilServantSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.official = {
            "uuid": "official-1",
            "occupation": "중앙정부 고위 공무원",
            "age": 44,
            "province": "대전",
            "district": "대전-유성구",
        }
        self.policy = {
            "상세정보": {
                "서비스명": "서울 청년 주거 지원",
                "지원대상": "서울 거주 만 20세 이상 39세 이하 청년",
                "지원내용": "월 20만 원을 최대 12개월 지원",
                "신청방법": "온라인 또는 주민센터 방문",
                "신청기한": "2026-09-30",
                "구비서류": "신분증과 거주 증명서",
                "제외조건": "주택 소유자 제외",
                "문의처": "서울 주거지원과 02-120",
            }
        }
        self.citizen = {
            "persona_id": "citizen-1",
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원으로는 주거 부담을 덜기 어렵습니다.",
                    "dialogue": "지원 규모가 현실적으로 부족합니다.",
                }
            ],
        }

    def test_support_response_uses_only_provided_policy_fact(self) -> None:
        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
        )

        self.assertEqual(result["official_persona_id"], "official-1")
        self.assertEqual(result["citizen_persona_id"], "citizen-1")
        self.assertIn("월 20만 원을 최대 12개월 지원", result["response"])
        self.assertNotIn("대전", result["response"])
        self.assertNotIn("승인하겠습니다", result["response"])
        self.assertEqual(result["_quality_gate"]["mode"], QUALITY_MODE)
        self.assertEqual(
            validate_civil_servant_response(
                result,
                persona=self.official,
                policy=self.policy,
                citizen_result=self.citizen,
            ),
            [],
        )

    def test_missing_contact_stays_unknown(self) -> None:
        policy = {
            "상세정보": {
                **self.policy["상세정보"],
                "문의처": "",
            }
        }
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "정보미제공",
                    "complaint_text": "문의처가 없어 어디에 물어볼지 모르겠습니다.",
                    "dialogue": "연락할 곳이 필요합니다.",
                }
            ],
        }

        _, response = build_grounded_response(policy, citizen)

        self.assertIn("문의처 정보가 명시되어 있지 않습니다", response)
        self.assertNotRegex(response, r"\d{2,4}-\d{3,4}-\d{4}")
        self.assertNotIn("http", response)

    def test_explicit_missing_field_beats_generic_application_phrase(self) -> None:
        policy = {
            "상세정보": {
                **self.policy["상세정보"],
                "문의처": "",
            }
        }
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "정보미제공",
                    "complaint_text": "문의처가 안내되지 않았습니다.",
                    "dialogue": "문의처가 없어 어떻게 신청해야 할지 모르겠습니다.",
                }
            ],
        }

        _, response = build_grounded_response(policy, citizen)

        self.assertIn("문의처 정보가 명시되어 있지 않습니다", response)
        self.assertNotIn("입력된 신청방법", response)

    def test_document_response_acknowledges_document_names(self) -> None:
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "서류 발급 절차가 복잡합니다.",
                    "dialogue": "제출 방법을 알고 싶습니다.",
                }
            ],
        }

        _, response = build_grounded_response(self.policy, citizen)

        self.assertIn("신분증과 거주 증명서", response)
        self.assertIn("구체적인 서류 발급처·유효기간·제출 순서", response)
        self.assertIn("온라인 또는 주민센터 방문", response)
        self.assertIn("서울 주거지원과 02-120", response)
        self.assertNotIn("3개월 이내", response)
        self.assertNotIn("복사본 금지", response)

    def test_english_policy_facts_are_displayed_in_korean_when_unambiguous(self) -> None:
        policy = {
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "KRW 200000 per month for up to 12 months",
                "신청기한": "2026-09-01 to 2026-09-30",
                "구비서류": "Identification card and proof of residence",
                "신청방법": "Apply online or visit local community service center",
                "문의처": "Seoul Housing Support Division 02-120",
            }
        }
        support_citizen = self.citizen
        deadline_citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "신청기한",
                    "complaint_text": "신청기한이 짧습니다.",
                    "dialogue": "준비 기간이 충분하지 않습니다.",
                }
            ],
        }

        _, support_response = build_grounded_response(policy, support_citizen)
        _, deadline_response = build_grounded_response(policy, deadline_citizen)

        self.assertIn("월 20만 원, 최대 12개월 지원", support_response)
        self.assertNotIn("KRW", support_response)
        self.assertIn("2026년 9월 1일부터 2026년 9월 30일까지", deadline_response)
        self.assertNotIn(" to ", deadline_response)

    def test_tampered_response_is_rejected(self) -> None:
        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
        )
        result["response"] += " 신청기한을 연장하겠습니다."

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertIn("OFFICIAL_RESPONSE_NOT_GROUNDED", errors)

    def test_wrong_citizen_pair_is_rejected(self) -> None:
        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
        )
        result["citizen_persona_id"] = "someone-else"

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertIn("OFFICIAL_CITIZEN_LINK_MISMATCH", errors)

    def test_inactive_official_is_rejected(self) -> None:
        inactive = {
            **self.official,
            "occupation": "전직 중앙정부 고위 공무원, 현재 구직중",
        }

        with self.assertRaisesRegex(RuntimeError, "OFFICIAL_PERSONA_NOT_ACTIVE"):
            run_civil_servant_simulation(inactive, self.policy, self.citizen)

    def test_exactly_one_validated_complaint_is_required(self) -> None:
        citizen = {**self.citizen, "complaints": []}

        with self.assertRaisesRegex(ValueError, "정확히 1개"):
            build_grounded_response(self.policy, citizen)

    def test_unknown_basis_fails_closed(self) -> None:
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "임의근거",
                    "complaint_text": "임의 민원",
                    "dialogue": "임의 대화",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "허용되지 않은"):
            build_grounded_response(self.policy, citizen)


if __name__ == "__main__":
    unittest.main()
