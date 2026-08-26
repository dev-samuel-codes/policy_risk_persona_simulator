import json
import unittest
from unittest.mock import patch

from backend.ai_simulation_core.prompts.citizen_prompt import citizen_prompt
from backend.ai_simulation_core.simulations.citizen_simulation import (
    parse_citizen_response,
    run_citizen_simulation,
    validate_citizen_response,
)
from backend.ai_simulation_core.simulations.citizen_quality import (
    MISSING_POLICY_VALUE,
    build_grounding_facts,
)


class CitizenSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.persona = {
            "uuid": "citizen-1",
            "age": 18,
            "occupation": "학생",
            "sex": "남성",
            "province": "서울",
            "district": "서울-서초구",
            "professional_persona": "자료를 꼼꼼하게 정리합니다.",
            "family_persona": "가족과 함께 거주합니다.",
            "persona": "지원 조건을 꼼꼼히 확인하는 학생입니다.",
        }
        self.policy = {
            "region_scope": "specific",
            "region_province": "서울",
            "region_district": "서울-서초구",
            "age_min": 19,
            "age_max": 34,
            "age_basis": "dataset_age",
            "상세정보": {
                "서비스명": "서울 청년 월세 지원",
                "지원대상": "청년",
                "선정기준": "",
                "지원내용": "월 20만 원 지원",
                "신청방법": "온라인",
                "신청기한": "상시",
                "구비서류": "",
                "제외조건": "",
            },
        }
        self.valid_result = {
            "persona_summary": {
                "이름": "김민준",
                "직업": "학생",
                "성별": "남성",
                "나이": "18",
                "거주지": "서울-서초구",
            },
            "grounding": build_grounding_facts(self.persona, self.policy),
            "personality": "지원 조건을 꼼꼼히 확인하는 학생입니다.",
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "나이 하한 때문에 지원 대상에서 제외됩니다.",
                    "dialogue": "한 살 차이로 지원을 못 받으니 답답합니다.",
                }
            ],
        }

    def test_prompt_includes_structured_region_and_age_conditions(self) -> None:
        prompt = citizen_prompt(self.persona, self.policy)

        self.assertIn("정책 적용 지역: 서울 / 서울-서초구", prompt)
        self.assertIn("[구조화된 정책 조건]", prompt)
        self.assertIn("정책 나이 조건: 19세 이상 34세 이하(경계 포함)", prompt)
        self.assertIn("나이 판단 기준: 페르소나 데이터셋의 정수 나이", prompt)
        self.assertIn("지역 조건 판정: 충족", prompt)
        self.assertIn("나이 조건 판정: 불충족", prompt)
        self.assertIn("전체 수급 자격과 실제 승인 여부: 입력만으로 확정 불가", prompt)
        self.assertIn(
            f"개인 주택 소유 여부: {MISSING_POLICY_VALUE}",
            prompt,
        )
        self.assertIn("무주택입니다", prompt)
        self.assertIn("지정된 쟁점의 민원 1개만", prompt)
        self.assertIn(f"구비서류: {MISSING_POLICY_VALUE}", prompt)
        self.assertIn("정보가 제공된 항목: 서비스명, 지원대상, 지원내용, 신청방법, 신청기한", prompt)
        self.assertIn("정보 미제공으로 비판 가능한 항목: 선정기준, 구비서류, 제외조건, 문의처", prompt)
        self.assertNotIn("소관기관명:", prompt)
        self.assertIn("[이번 응답의 유일한 민원 쟁점 - 변경 금지]", prompt)
        self.assertIn("complaints는 정확히 1개인 배열", prompt)
        self.assertIn("- basis: 정보미제공", prompt)
        self.assertIn('"basis": "정보미제공"', prompt)
        self.assertIn("나이·지역·지원 대상·자격·승인·제외", prompt)
        self.assertIn("조건 충족 설명을 모두 쓰지 말 것", prompt)
        self.assertNotIn("온라인으로만 가능", prompt)
        self.assertNotIn("컴퓨터를 잘 모르는 사람", prompt)

    def test_complete_policy_uses_one_safe_provided_field_focus(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증과 주민등록등본",
                "제외조건": "주택 소유자 제외",
                "문의처": "서울 주거지원과 02-120",
            },
        }

        prompt = citizen_prompt(self.persona, policy)
        safe_bases = ("지원내용", "신청방법", "신청기한", "구비서류")

        self.assertTrue(any(f"- basis: {basis}" in prompt for basis in safe_bases))
        self.assertNotIn("- basis: 선정기준", prompt)
        self.assertNotIn("- basis: 제외조건", prompt)
        self.assertIn("complaints는 정확히 1개인 배열", prompt)

    def test_document_focus_acknowledges_provided_document_names(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증과 주민등록등본",
                "제외조건": "주택 소유자 제외",
                "문의처": "서울 주거지원과 02-120",
            },
        }
        persona = {**self.persona, "uuid": "c"}

        prompt = citizen_prompt(persona, policy)

        self.assertIn("- basis: 구비서류", prompt)
        self.assertIn("제공된 구비서류 이름을 인정한 상태", prompt)
        self.assertIn("서류 항목 자체가 없거나 어떤 서류인지 모른다고 말하지 말 것", prompt)

    def test_missing_exclusion_alone_is_not_selected_as_information_gap(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증과 주민등록등본",
                "제외조건": "",
                "문의처": "서울 주거지원과 02-120",
            },
        }

        prompt = citizen_prompt(self.persona, policy)

        self.assertNotIn("- basis: 제외조건", prompt)
        self.assertNotIn("제외조건 정보가 제공되지 않은 점만 지적", prompt)
        self.assertTrue(
            any(
                f"- basis: {basis}" in prompt
                for basis in ("지원내용", "신청방법", "신청기한", "구비서류")
            )
        )

    def test_missing_contact_template_uses_correct_particles(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증",
                "문의처": "",
            },
        }

        prompt = citizen_prompt(self.persona, policy)

        self.assertIn("문의처가 안내되지 않아", prompt)
        self.assertIn("문의처를 어디서 확인할 수 있는지", prompt)
        self.assertNotIn("문의처이", prompt)
        self.assertNotIn("문의처을", prompt)

    def test_prompt_distinguishes_unrestricted_and_legacy_conditions(self) -> None:
        unrestricted = {
            **self.policy,
            "region_scope": "nationwide",
            "region_province": "",
            "region_district": "",
            "age_min": None,
            "age_max": None,
        }
        legacy = {"상세정보": self.policy["상세정보"]}

        unrestricted_prompt = citizen_prompt(self.persona, unrestricted)
        legacy_prompt = citizen_prompt(self.persona, legacy)

        self.assertIn("정책 적용 지역: 전국", unrestricted_prompt)
        self.assertIn("정책 나이 조건: 제한 없음", unrestricted_prompt)
        self.assertIn("구조화 조건 없음", legacy_prompt)

    def test_validation_requires_complete_complaint_contract(self) -> None:
        invalid = {
            **self.valid_result,
            "persona_summary": {"이름": "김민준", "나이": ""},
            "personality": "",
            "complaints": [{"complaint_text": "", "dialogue": ""}],
        }

        errors = validate_citizen_response(invalid, self.persona)

        self.assertTrue(any("나이 필드" in error for error in errors))
        self.assertTrue(any("personality" in error for error in errors))
        self.assertTrue(any("complaint_text" in error for error in errors))
        self.assertTrue(any("dialogue" in error for error in errors))

    def test_validation_requires_exactly_one_complaint(self) -> None:
        no_complaints = {**self.valid_result, "complaints": []}
        two_complaints = {
            **self.valid_result,
            "complaints": self.valid_result["complaints"] * 2,
        }

        self.assertIn(
            "complaints는 정확히 1개여야 함",
            validate_citizen_response(no_complaints, self.persona),
        )
        self.assertIn(
            "complaints는 정확히 1개여야 함",
            validate_citizen_response(two_complaints, self.persona),
        )

    def test_locked_persona_summary_may_keep_english_occupation(self) -> None:
        persona = {**self.persona, "occupation": "ICT 강사"}
        result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "직업": "ICT 강사",
            },
        }

        errors = validate_citizen_response(result, persona)

        self.assertNotIn("영어 단어 혼용 감지", errors)

    def test_generated_name_and_short_english_terms_are_rejected(self) -> None:
        english_name = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "이름": "Alex",
            },
        }
        short_english = {
            **self.valid_result,
            "personality": "AI와 IT 활용에 익숙합니다.",
        }

        self.assertIn(
            "영어 단어 혼용 감지",
            validate_citizen_response(english_name, self.persona),
        )
        self.assertIn(
            "영어 단어 혼용 감지",
            validate_citizen_response(short_english, self.persona),
        )

    def test_parser_normalizes_equivalent_complaint_basis_labels(self) -> None:
        parsed = parse_citizen_response(
            json.dumps(
                {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "신청 절차",
                            "complaint_text": "신청 절차가 복잡할 수 있습니다.",
                            "dialogue": "신청 절차를 더 쉽게 설명해 주세요.",
                        },
                        {
                            "basis": "제출 서류",
                            "complaint_text": "제출 서류 준비가 부담됩니다.",
                            "dialogue": "필요한 서류를 쉽게 확인하고 싶습니다.",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(
            [complaint["basis"] for complaint in parsed["complaints"]],
            ["신청방법", "구비서류"],
        )

    def test_parser_does_not_guess_unknown_complaint_basis(self) -> None:
        parsed = parse_citizen_response(
            json.dumps(
                {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "디지털격차",
                            "complaint_text": "온라인 이용이 어려울 수 있습니다.",
                            "dialogue": "방문 지원도 필요합니다.",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(parsed["complaints"][0]["basis"], "디지털격차")
        errors = validate_citizen_response(parsed, self.persona, self.policy)
        self.assertTrue(any(error.startswith("COMPLAINT_BASIS_INVALID") for error in errors))

    def test_parser_maps_missing_agency_field_to_information_gap(self) -> None:
        parsed = parse_citizen_response(
            json.dumps(
                {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "소관기관명",
                            "complaint_text": "소관기관명이 제공되지 않았습니다.",
                            "dialogue": "어느 기관이 담당하는지 알고 싶습니다.",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(parsed["complaints"][0]["basis"], "정보미제공")

    def test_prompt_preserves_zero_and_one_sided_age_boundaries(self) -> None:
        minimum_only = {
            **self.policy,
            "age_min": 0,
            "age_max": None,
        }
        maximum_only = {
            **self.policy,
            "age_min": None,
            "age_max": 34,
        }

        self.assertIn("정책 나이 조건: 0세 이상", citizen_prompt(self.persona, minimum_only))
        self.assertIn(
            "정책 나이 조건: 34세 이하",
            citizen_prompt(self.persona, maximum_only),
        )

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_invalid_results_fail_closed_after_all_retries(self, run_llm) -> None:
        invalid = {**self.valid_result, "complaints": []}
        run_llm.return_value = json.dumps(invalid, ensure_ascii=False)

        result = run_citizen_simulation(
            self.persona,
            self.policy,
            max_retries=3,
        )

        self.assertIsNone(result)
        self.assertEqual(run_llm.call_count, 3)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_retry_returns_only_a_valid_result(self, run_llm) -> None:
        invalid = {**self.valid_result, "complaints": []}
        run_llm.side_effect = [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(self.valid_result, ensure_ascii=False),
        ]

        result = run_citizen_simulation(
            self.persona,
            self.policy,
            max_retries=3,
        )

        self.assertEqual(run_llm.call_count, 2)
        self.assertEqual(result["persona_id"], "citizen-1")
        self.assertEqual(result["_validation_errors"], [])
        self.assertEqual(result["_quality_gate"]["status"], "passed")
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 2)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_maximum_support_shorthand_is_normalized_before_validation(
        self,
        run_llm,
    ) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "월 최대 20만 원을 최대 12개월 지원",
            },
        }
        generated = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "제공된 지원 규모가 충분한지 의문입니다.",
                    "dialogue": "월 20만 원을 최대 12개월 지원하는 규모가 충분한가요?",
                }
            ],
        }
        run_llm.return_value = json.dumps(generated, ensure_ascii=False)

        result = run_citizen_simulation(self.persona, policy, max_retries=3)

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 1)
        self.assertIn("월 최대 20만 원", result["complaints"][0]["dialogue"])
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_semantic_failure_feedback_is_used_on_retry(self, run_llm) -> None:
        eligible_persona = {**self.persona, "age": 25}
        valid = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, self.policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "지원 규모가 충분한지 궁금합니다.",
                    "dialogue": "지원 내용이 제 상황에 충분한지 더 알고 싶습니다.",
                }
            ],
        }
        invalid = {
            **valid,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "제 나이 때문에 지원 대상에서 제외됩니다.",
                    "dialogue": "스물다섯 살인데 나이 때문에 지원을 못 받습니다.",
                }
            ],
        }
        run_llm.side_effect = [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(valid, ensure_ascii=False),
        ]

        result = run_citizen_simulation(
            eligible_persona,
            self.policy,
            max_retries=3,
        )

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 2)
        retry_prompt = run_llm.call_args_list[1].args[0]
        self.assertIn("AGE_ELIGIBILITY_CONTRADICTION", retry_prompt)
        self.assertIn("나이 조건을 충족하므로", retry_prompt)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_single_invalid_complaint_triggers_full_retry(self, run_llm) -> None:
        eligible_persona = {**self.persona, "age": 25}
        mixed_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, self.policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원 지원이 충분한지 걱정됩니다.",
                    "dialogue": "지원 규모의 실효성을 더 설명해 주세요.",
                },
                {
                    "basis": "신청방법",
                    "complaint_text": "온라인 신청 절차가 복잡할까 걱정됩니다.",
                    "dialogue": "온라인 신청 단계를 쉽게 안내해 주세요.",
                },
                {
                    "basis": "지원대상",
                    "complaint_text": "나이와 지역 조건은 충족합니다.",
                    "dialogue": "조건에 맞는데 왜 지원이 안 되는지 모르겠어요.",
                },
            ],
        }
        valid_result = {
            **mixed_result,
            "complaints": mixed_result["complaints"][:1],
        }
        run_llm.side_effect = [
            json.dumps(mixed_result, ensure_ascii=False),
            json.dumps(valid_result, ensure_ascii=False),
        ]

        result = run_citizen_simulation(
            eligible_persona,
            self.policy,
            max_retries=3,
        )

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 2)
        self.assertEqual(len(result["complaints"]), 1)
        self.assertEqual(result["complaints"][0]["basis"], "지원내용")
        self.assertEqual(result["_quality_gate"]["removed_complaints"], 0)
        retry_prompt = run_llm.call_args_list[1].args[0]
        self.assertIn("UNSUPPORTED_CURRENT_OUTCOME", retry_prompt)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_single_invalid_complaint_retries_when_only_one_would_remain(
        self,
        run_llm,
    ) -> None:
        eligible_persona = {**self.persona, "age": 25}
        valid_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, self.policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원 지원이 충분한지 걱정됩니다.",
                    "dialogue": "지원 규모의 실효성을 더 설명해 주세요.",
                }
            ],
        }
        one_valid_one_invalid = {
            **valid_result,
            "complaints": [
                *valid_result["complaints"],
                {
                    "basis": "지원대상",
                    "complaint_text": "나이와 지역 조건은 충족합니다.",
                    "dialogue": "조건에 맞는데 왜 지원이 안 되는지 모르겠어요.",
                },
            ],
        }
        run_llm.side_effect = [
            json.dumps(one_valid_one_invalid, ensure_ascii=False),
            json.dumps(valid_result, ensure_ascii=False),
        ]

        result = run_citizen_simulation(
            eligible_persona,
            self.policy,
            max_retries=3,
        )

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 2)
        self.assertEqual(len(result["complaints"]), 1)
        self.assertEqual(result["_quality_gate"]["removed_complaints"], 0)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_multiple_invalid_complaints_trigger_retry(self, run_llm) -> None:
        eligible_persona = {**self.persona, "age": 25}
        valid_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, self.policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원 지원이 충분한지 걱정됩니다.",
                    "dialogue": "지원 규모의 실효성을 더 설명해 주세요.",
                }
            ],
        }
        mostly_invalid_result = {
            **valid_result,
            "complaints": [
                *valid_result["complaints"],
                {
                    "basis": "지원대상",
                    "complaint_text": "제 나이 때문에 지원 대상에서 제외됩니다.",
                    "dialogue": "스물다섯 살인데 나이 때문에 지원을 못 받습니다.",
                },
                {
                    "basis": "지원대상",
                    "complaint_text": "서울 거주자지만 지역 때문에 제외됩니다.",
                    "dialogue": "서울에 살아도 지역 조건 때문에 지원을 못 받습니다.",
                },
            ],
        }
        run_llm.side_effect = [
            json.dumps(mostly_invalid_result, ensure_ascii=False),
            json.dumps(valid_result, ensure_ascii=False),
        ]

        result = run_citizen_simulation(
            eligible_persona,
            self.policy,
            max_retries=3,
        )

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 2)
        self.assertEqual(len(result["complaints"]), 1)
        self.assertEqual(result["_quality_gate"]["removed_complaints"], 0)
        retry_prompt = run_llm.call_args_list[1].args[0]
        self.assertIn("AGE_ELIGIBILITY_CONTRADICTION", retry_prompt)
        self.assertIn("REGION_ELIGIBILITY_CONTRADICTION", retry_prompt)

    def test_retry_feedback_explains_provided_policy_fact_in_korean(self) -> None:
        prompt = citizen_prompt(
            self.persona,
            self.policy,
            validation_feedback=[
                "CONTRADICTED_POLICY_FACT:지원대상:complaints[0].dialogue"
            ],
        )

        self.assertIn("지원대상은 정책 사실 블록에 이미 제공되었습니다", prompt)
        self.assertIn("없거나 알 수 없다고 쓰지 말고", prompt)

    def test_retry_feedback_explains_current_denial_presupposition(self) -> None:
        prompt = citizen_prompt(
            self.persona,
            self.policy,
            validation_feedback=[
                "UNSUPPORTED_CURRENT_OUTCOME:complaints[0].dialogue"
            ],
        )

        self.assertIn("현재 지원이 거절되었다는 전제를 삭제하세요", prompt)
        self.assertIn("다른 조건이 있는지 확인이 필요하다", prompt)

    def test_retry_feedback_removes_all_eligibility_preamble(self) -> None:
        prompt = citizen_prompt(
            self.persona,
            self.policy,
            validation_feedback=[
                "UNSUPPORTED_OVERALL_ELIGIBILITY:complaints[0].dialogue"
            ],
        )

        self.assertIn("나이·지역·지원 대상·자격 언급", prompt)
        self.assertIn("전부 삭제하세요", prompt)
        self.assertIn("지정된 단일 쟁점만 남기세요", prompt)

    def test_retry_feedback_translates_source_alphabet_to_korean(self) -> None:
        prompt = citizen_prompt(
            self.persona,
            self.policy,
            validation_feedback=["영어 단어 혼용 감지"],
        )

        self.assertIn("이름·성격·민원 본문·대화 문장의 알파벳을 모두 삭제하세요", prompt)
        self.assertIn("영문 약어가 있어도 그대로 복사하지 말고", prompt)
        self.assertNotIn("ICT는 정보통신기술", prompt)

    def test_prompt_koreanizes_persona_source_alphabet(self) -> None:
        persona = {
            **self.persona,
            "professional_persona": "ICT 전공자로 SNS 활용 교육을 합니다.",
            "family_persona": "YouTube 영상을 가족과 봅니다.",
            "persona": "PowerPoint와 Excel을 사용합니다.",
        }

        prompt = citizen_prompt(persona, self.policy)

        self.assertIn("정보통신기술 전공자로 사회관계망서비스 활용 교육", prompt)
        self.assertIn("유튜브 영상을 가족과 봅니다", prompt)
        self.assertIn("발표자료와 엑셀을 사용합니다", prompt)
        self.assertNotIn("ICT 전공자로", prompt)
        self.assertNotIn("SNS 활용 교육", prompt)

    def test_prompt_locks_source_persona_name_when_present(self) -> None:
        persona = {
            **self.persona,
            "professional_persona": "이혜진 씨는 꼼꼼한 상담원입니다.",
        }

        prompt = citizen_prompt(persona, self.policy)

        self.assertIn('"이름": "이혜진"', prompt)
        self.assertIn("이름·나이·직업·성별·거주지는 출력 계약", prompt)

    def test_prompt_cleans_unknown_source_alphabet_punctuation(self) -> None:
        persona = {
            **self.persona,
            "professional_persona": "Rust/Go와 (Kotlin) 도구를 사용합니다.",
        }

        prompt = citizen_prompt(persona, self.policy)

        self.assertNotIn("Rust/Go", prompt)
        self.assertNotIn("Kotlin", prompt)
        self.assertNotIn("()", prompt)

    def test_retry_feedback_gives_safe_housing_rewrite(self) -> None:
        prompt = citizen_prompt(
            self.persona,
            self.policy,
            validation_feedback=[
                "UNSUPPORTED_PERSONA_FACT:주거:complaints[0].dialogue"
            ],
        )

        self.assertIn("1인칭 주거 단정을 전부 빼세요", prompt)
        self.assertIn("주택 소유 여부에 따른 적용 기준", prompt)


if __name__ == "__main__":
    unittest.main()
