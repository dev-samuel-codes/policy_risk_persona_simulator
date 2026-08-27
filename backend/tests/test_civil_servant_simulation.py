import json
import unittest
from unittest.mock import patch

from backend.ai_simulation_core.prompts.civil_servant_prompt import (
    civil_servant_prompt,
)
from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    QUALITY_MODE,
    build_grounded_response,
    parse_civil_servant_response,
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
            "persona": "근거를 확인한 뒤 신중하게 안내합니다.",
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
            "persona": {
                "uuid": "citizen-1",
                "age": 29,
                "province": "서울",
                "district": "서울-강서구",
            },
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원으로는 주거 부담을 덜기 어렵습니다.",
                    "dialogue": "지원 규모가 현실적으로 부족합니다.",
                }
            ],
        }
        self.safe_response = (
            "민원 내용을 확인했습니다. 지원 규모의 실효성은 검토할 필요가 있습니다. "
            "추가 지원 여부는 입력된 정책만으로 확정할 수 없어 공식 확인이 필요합니다."
        )

    def _result(self, response: str | None = None) -> dict:
        return {
            "official_persona_id": "official-1",
            "citizen_persona_id": "citizen-1",
            "basis": "지원내용",
            "response": response or self.safe_response,
            "_validation_errors": [],
            "_quality_gate": {
                "status": "passed",
                "mode": QUALITY_MODE,
                "removed_statements": 0,
                "generation_attempts": 1,
            },
        }

    def _llm_output(self, response: str | None = None) -> str:
        return json.dumps(
            {"response": response or self.safe_response},
            ensure_ascii=False,
        )

    def test_prompt_contains_official_policy_and_single_citizen_complaint(
        self,
    ) -> None:
        _, grounded_response = build_grounded_response(self.policy, self.citizen)
        complaint = self.citizen["complaints"][0]["complaint_text"]

        prompt = civil_servant_prompt(
            self.official,
            self.policy,
            complaint,
            grounded_response,
        )

        self.assertIn('"uuid": "official-1"', prompt)
        self.assertIn('"서비스명": "서울 청년 주거 지원"', prompt)
        self.assertEqual(prompt.count(complaint), 1)
        self.assertIn(grounded_response, prompt)
        self.assertIn('"response"', prompt)
        self.assertNotIn("정책 집행 리스크", prompt)

    def test_parser_accepts_json_response(self) -> None:
        parsed = parse_civil_servant_response(self._llm_output())

        self.assertEqual(parsed, self.safe_response)

    def test_parser_accepts_json_code_fence(self) -> None:
        fence = chr(96) * 3
        raw = f"{fence}json\n{self._llm_output()}\n{fence}"

        parsed = parse_civil_servant_response(raw)

        self.assertEqual(parsed, self.safe_response)

    def test_parser_accepts_plain_text_response(self) -> None:
        parsed = parse_civil_servant_response(self.safe_response)

        self.assertEqual(parsed, self.safe_response)

    def test_parser_rejects_malformed_json(self) -> None:
        parsed = parse_civil_servant_response('{"response": "닫히지 않은 응답"')

        self.assertIsNone(parsed)

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_qwen_response_is_wrapped_in_server_owned_contract(self, run_llm) -> None:
        run_llm.return_value = self._llm_output()

        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
            max_retries=3,
        )

        _, deterministic_response = build_grounded_response(self.policy, self.citizen)
        self.assertEqual(run_llm.call_count, 1)
        self.assertEqual(result["official_persona_id"], "official-1")
        self.assertEqual(result["citizen_persona_id"], "citizen-1")
        self.assertEqual(result["basis"], "지원내용")
        self.assertEqual(result["response"], self.safe_response)
        self.assertNotEqual(result["response"], deterministic_response)
        self.assertEqual(result["_validation_errors"], [])
        self.assertEqual(result["_quality_gate"]["status"], "passed")
        self.assertEqual(result["_quality_gate"]["mode"], QUALITY_MODE)
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)
        self.assertEqual(QUALITY_MODE, "qwen_best_effort_v1")
        self.assertEqual(
            validate_civil_servant_response(
                result,
                persona=self.official,
                policy=self.policy,
                citizen_result=self.citizen,
            ),
            [],
        )

    def test_policy_external_numeric_fact_is_rejected(self) -> None:
        result = self._result("입력된 정책에 따라 월 100만 원을 지원합니다.")

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_NUMERIC_FACT:") for error in errors)
        )

    def test_unapproved_official_commitment_is_rejected(self) -> None:
        result = self._result("신청기한을 연장하겠습니다.")

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertIn("UNSUPPORTED_OFFICIAL_COMMITMENT", errors)

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_runtime_accepts_first_nonempty_response_without_quality_retry(
        self,
        run_llm,
    ) -> None:
        generated_response = "국토교통부 청년정책과가 이 사업의 담당 기관입니다."
        run_llm.return_value = self._llm_output(generated_response)

        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
            max_retries=3,
        )

        self.assertEqual(run_llm.call_count, 1)
        self.assertEqual(result["response"], generated_response)
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)
        self.assertFalse(result["_quality_gate"]["fallback_used"])

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_qwen_error_uses_grounded_fallback_without_retry(self, run_llm) -> None:
        run_llm.side_effect = RuntimeError("Qwen unavailable")

        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
            max_retries=3,
        )
        _, fallback_response = build_grounded_response(self.policy, self.citizen)

        self.assertEqual(run_llm.call_count, 1)
        self.assertEqual(result["response"], fallback_response)
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)
        self.assertTrue(result["_quality_gate"]["fallback_used"])

    def test_missing_contact_stays_unknown_in_grounded_draft(self) -> None:
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

    def test_document_draft_acknowledges_document_names(self) -> None:
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

    def test_english_policy_facts_are_koreanized_in_grounded_draft(self) -> None:
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

        _, support_response = build_grounded_response(policy, self.citizen)
        _, deadline_response = build_grounded_response(policy, deadline_citizen)

        self.assertIn("월 20만 원, 최대 12개월 지원", support_response)
        self.assertNotIn("KRW", support_response)
        self.assertIn("2026년 9월 1일부터 2026년 9월 30일까지", deadline_response)
        self.assertNotIn(" to ", deadline_response)

    def test_generated_contact_details_and_agency_are_rejected(self) -> None:
        cases = (
            (
                "전화번호",
                "자세한 내용은 02-1234-5678로 문의하세요.",
                "UNSUPPORTED_CONTACT_FACT:PHONE:official.response",
            ),
            (
                "웹주소",
                "https://youth.example.com에서 신청 내용을 확인하세요.",
                "UNSUPPORTED_CONTACT_FACT:URL:official.response",
            ),
            (
                "이메일",
                "housing@example.com으로 문의하세요.",
                "UNSUPPORTED_CONTACT_FACT:EMAIL:official.response",
            ),
            (
                "담당기관",
                "국토교통부 청년정책과가 이 사업의 담당 기관입니다.",
                "UNSUPPORTED_CONTACT_FACT:AGENCY:official.response",
            ),
        )

        for label, response, expected_error in cases:
            with self.subTest(contact_kind=label):
                errors = validate_civil_servant_response(
                    self._result(response),
                    persona=self.official,
                    policy=self.policy,
                    citizen_result=self.citizen,
                )
                self.assertIn(expected_error, errors)

    def test_contact_detail_present_in_policy_is_allowed(self) -> None:
        response = (
            "지원 내용에 관한 입력 정책의 문의처는 서울 주거지원과 02-120입니다. "
            "추가 지원 여부는 해당 문의처의 공식 안내 확인이 필요합니다."
        )

        errors = validate_civil_servant_response(
            self._result(response),
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertFalse(
            any(error.startswith("UNSUPPORTED_CONTACT_FACT:") for error in errors),
            errors,
        )

    def test_generic_responsible_agency_reference_is_allowed(self) -> None:
        response = (
            "지원 내용의 세부 사항은 해당 사업의 담당 기관에 확인이 필요합니다."
        )

        errors = validate_civil_servant_response(
            self._result(response),
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertNotIn(
            "UNSUPPORTED_CONTACT_FACT:AGENCY:official.response",
            errors,
        )

    def test_parser_rejects_explanation_or_think_block_mixed_with_json(self) -> None:
        fence = chr(96) * 3
        cases = (
            f"설명:\n{fence}json\n{self._llm_output()}\n{fence}",
            f"<think>정책 근거를 검토합니다.</think>\n{self._llm_output()}",
        )

        for raw_output in cases:
            with self.subTest(raw_output=raw_output):
                self.assertIsNone(parse_civil_servant_response(raw_output))

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_mixed_output_uses_grounded_fallback_without_retry(self, run_llm) -> None:
        fence = chr(96) * 3
        run_llm.return_value = (
            f"추가 설명\n{fence}json\n{self._llm_output()}\n{fence}"
        )

        result = run_civil_servant_simulation(
            self.official,
            self.policy,
            self.citizen,
            max_retries=3,
        )
        _, fallback_response = build_grounded_response(self.policy, self.citizen)

        self.assertEqual(run_llm.call_count, 1)
        self.assertEqual(result["response"], fallback_response)
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)
        self.assertTrue(result["_quality_gate"]["fallback_used"])

    def test_uncertainty_does_not_excuse_invented_government24_channel(self) -> None:
        policy = {
            "상세정보": {
                **self.policy["상세정보"],
                "신청방법": "",
            }
        }
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "신청방법",
                    "complaint_text": "신청 경로를 확인하기 어렵습니다.",
                    "dialogue": "어디에서 신청해야 하는지 안내해 주세요.",
                }
            ],
        }
        result = self._result(
            "정확한 신청 경로는 확인이 필요하지만 정부24로 신청하세요."
        )
        result["basis"] = "신청방법"

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=policy,
            citizen_result=citizen,
        )

        self.assertIn(
            "UNSUPPORTED_POLICY_FACT:신청방법:official.response:government24",
            errors,
        )

    def test_generic_application_documents_are_not_specific_document_fact(self) -> None:
        policy = {
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "",
            }
        }
        citizen = {
            **self.citizen,
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "필요한 서류가 안내되지 않았습니다.",
                    "dialogue": "어떤 서류를 준비해야 하나요?",
                }
            ],
        }
        result = self._result("신청서류는 공식 공고를 확인해 주세요.")
        result["basis"] = "구비서류"

        errors = validate_civil_servant_response(
            result,
            persona=self.official,
            policy=policy,
            citizen_result=citizen,
        )

        self.assertFalse(
            any(
                error.startswith("UNSUPPORTED_POLICY_FACT:구비서류:")
                for error in errors
            ),
            errors,
        )

    def test_response_must_address_the_validated_complaint_basis(self) -> None:
        response = (
            "입력된 신청기한은 2026-09-30입니다. "
            "기한 변경 여부는 공식 공고 확인이 필요합니다."
        )

        errors = validate_civil_servant_response(
            self._result(response),
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertIn("OFFICIAL_BASIS_UNADDRESSED", errors)

    def test_risk_scoring_language_is_rejected(self) -> None:
        cases = (
            "이 민원의 위험도는 높습니다.",
            "정책 집행 리스크가 큽니다.",
            "민원 점수는 낮음입니다.",
        )

        for response in cases:
            with self.subTest(response=response):
                errors = validate_civil_servant_response(
                    self._result(response),
                    persona=self.official,
                    policy=self.policy,
                    citizen_result=self.citizen,
                )
                self.assertIn("OFFICIAL_RISK_SCORING_CONTENT", errors)

    def test_general_administrative_support_is_not_a_commitment(self) -> None:
        response = (
            "민원에서 제기한 지원 내용의 안내를 지원하겠습니다. "
            "추가 지원 여부는 공식 확인이 필요합니다."
        )

        errors = validate_civil_servant_response(
            self._result(response),
            persona=self.official,
            policy=self.policy,
            citizen_result=self.citizen,
        )

        self.assertNotIn("UNSUPPORTED_OFFICIAL_COMMITMENT", errors)

    def test_actual_approval_payment_and_eligibility_promises_are_rejected(
        self,
    ) -> None:
        promises = (
            "신청을 승인하겠습니다.",
            "지원금을 지급하겠습니다.",
            "지원 대상자로 처리해 드리겠습니다.",
            "신청기한을 연장하겠습니다.",
        )

        for response in promises:
            with self.subTest(response=response):
                errors = validate_civil_servant_response(
                    self._result(response),
                    persona=self.official,
                    policy=self.policy,
                    citizen_result=self.citizen,
                )
                self.assertIn("UNSUPPORTED_OFFICIAL_COMMITMENT", errors)

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_wrong_citizen_pair_is_rejected(self, run_llm) -> None:
        run_llm.return_value = self._llm_output()
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

    @patch(
        "backend.ai_simulation_core.simulations.civil_servant_simulation.run_llm"
    )
    def test_inactive_official_is_rejected(self, run_llm) -> None:
        run_llm.return_value = self._llm_output()
        inactive = {
            **self.official,
            "occupation": "전직 중앙정부 고위 공무원, 현재 구직중",
        }

        with self.assertRaisesRegex(RuntimeError, "OFFICIAL_PERSONA_NOT_ACTIVE"):
            run_civil_servant_simulation(inactive, self.policy, self.citizen)

    def test_exactly_one_validated_complaint_is_required(self) -> None:
        for complaints in ([], self.citizen["complaints"] * 2):
            with self.subTest(complaint_count=len(complaints)):
                citizen = {**self.citizen, "complaints": complaints}
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
