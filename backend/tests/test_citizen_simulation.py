import json
import unittest
from unittest.mock import patch

from backend.ai_simulation_core.prompts.citizen_prompt import (
    ComplaintFocus,
    _complaint_focus,
    citizen_prompt,
)
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
                    "complaint_text": (
                        "현재 18세로 정책의 19세 이상 연령 하한보다 한 살 낮습니다."
                    ),
                    "dialogue": (
                        "저는 현재 18세이고 정책은 19세 이상인데, 한 살 차이 "
                        "기준의 이유와 적용 방식을 설명해 주세요."
                    ),
                }
            ],
        }

    def complete_policy(self, **detail_changes) -> dict:
        return {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "선정기준": "정책 원문에 따른 심사",
                "구비서류": "신분증과 주민등록등본",
                "제외조건": "별도 제외조건 없음",
                "문의처": "서울 주거지원과 02-120",
                **detail_changes,
            },
        }

    def test_age_boundary_focus_precedes_missing_policy_fields(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)

        self.assertIsInstance(focus, ComplaintFocus)
        self.assertEqual(focus.kind, "age_boundary")
        self.assertEqual(focus.basis, "지원대상")
        self.assertEqual(focus.priority, 100)
        self.assertEqual(focus.policy_evidence, "연령 하한은 19세 이상")
        self.assertEqual(focus.persona_evidence, "현재 18세로 하한보다 한 살 낮음")

    def test_upper_age_boundary_focus_uses_policy_and_persona_evidence(self) -> None:
        persona = {**self.persona, "age": 35}

        focus = _complaint_focus(persona, self.policy)

        self.assertEqual(focus.kind, "age_boundary")
        self.assertEqual(focus.policy_evidence, "연령 상한은 34세 이하")
        self.assertEqual(focus.persona_evidence, "현재 35세로 상한보다 한 살 높음")

    def test_region_mismatch_focus_precedes_information_gaps(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "province": "부산",
            "district": "부산-해운대구",
        }

        focus = _complaint_focus(persona, self.policy)

        self.assertEqual(focus.kind, "region_mismatch")
        self.assertEqual(focus.basis, "지원대상")
        self.assertEqual(focus.priority, 95)
        self.assertIn("서울 / 서울-서초구", focus.policy_evidence)
        self.assertIn("부산-해운대구", focus.persona_evidence)

    def test_region_focus_uses_the_shared_district_normalization(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = {
            **self.complete_policy(),
            "region_district": "서초구",
        }

        same_region_focus = _complaint_focus(persona, policy)
        different_region_focus = _complaint_focus(
            {**persona, "district": "서울-강남구"},
            policy,
        )

        self.assertNotEqual(same_region_focus.kind, "region_mismatch")
        self.assertEqual(different_region_focus.kind, "region_mismatch")

    def test_age_boundary_precedes_simultaneous_region_mismatch(self) -> None:
        persona = {
            **self.persona,
            "province": "부산",
            "district": "부산-해운대구",
        }

        focus = _complaint_focus(persona, self.policy)

        self.assertEqual(focus.kind, "age_boundary")
        self.assertEqual(focus.priority, 100)

    def test_missing_target_preserves_known_structured_conditions(self) -> None:
        policy = self.complete_policy(지원대상="")
        persona = {**self.persona, "age": 25}

        focus = _complaint_focus(persona, policy)

        self.assertEqual(focus.kind, "eligibility_gap")
        self.assertEqual(focus.priority, 80)
        self.assertIn("별도 지원대상 항목이 비어", focus.policy_evidence)
        self.assertIn("다른 정책 항목", focus.instruction)

    def test_missing_selection_criteria_precedes_administrative_gaps(self) -> None:
        policy = self.complete_policy(선정기준="", 문의처="")
        persona = {**self.persona, "age": 25}

        focus = _complaint_focus(persona, policy)

        self.assertEqual(focus.kind, "selection_criteria_gap")
        self.assertEqual(focus.basis, "선정기준")
        self.assertEqual(focus.priority, 80)

    def test_blank_field_gap_keeps_other_fields_as_known_context(
        self,
    ) -> None:
        persona = {**self.persona, "age": 25}
        cases = (
            (
                "eligibility_gap",
                self.complete_policy(
                    지원대상="",
                    선정기준="서울 청년 중 중위소득 60퍼센트 이하인 사람",
                ),
            ),
            (
                "selection_criteria_gap",
                self.complete_policy(
                    지원대상="서울 청년 중 중위소득 60퍼센트 이하를 소득순 우선 선정",
                    선정기준="",
                ),
            ),
            (
                "exclusion_gap",
                self.complete_policy(
                    지원대상="서울 청년 중 주택 소유자 제외",
                    제외조건="",
                ),
            ),
        )

        for expected_kind, policy in cases:
            with self.subTest(kind=expected_kind):
                focus = _complaint_focus(persona, policy)
                self.assertEqual(focus.kind, expected_kind)
                self.assertIn("다른 정책 항목", focus.policy_evidence)
                self.assertIn("다른 정책 항목", focus.instruction)

    def test_safe_focuses_only_use_fields_with_values(self) -> None:
        policy = self.complete_policy(지원내용="")
        persona = {**self.persona, "uuid": "d", "age": 25}

        focus = _complaint_focus(persona, policy)

        self.assertNotEqual(focus.basis, "지원내용")
        self.assertNotIn(MISSING_POLICY_VALUE, focus.policy_evidence)

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
        self.assertIn("- 유형: age_boundary", prompt)
        self.assertIn("- basis: 지원대상", prompt)
        self.assertIn('"basis": "지원대상"', prompt)
        self.assertIn("- 정책 근거: 연령 하한은 19세 이상", prompt)
        self.assertIn("- 페르소나 근거: 현재 18세로 하한보다 한 살 낮음", prompt)
        self.assertNotIn("구비서류 정보가 제공되지 않은 점만 지적", prompt)
        self.assertNotIn("- 유형: missing_구비서류", prompt)
        self.assertNotIn("온라인으로만 가능", prompt)
        self.assertNotIn("컴퓨터를 잘 모르는 사람", prompt)

    def test_complete_policy_uses_one_safe_provided_field_focus(self) -> None:
        policy = self.complete_policy(제외조건="주택 소유자 제외")
        persona = {**self.persona, "age": 25}

        prompt = citizen_prompt(persona, policy)
        safe_bases = ("지원내용", "신청방법", "신청기한", "구비서류")

        self.assertTrue(any(f"- basis: {basis}" in prompt for basis in safe_bases))
        self.assertNotIn("- basis: 선정기준", prompt)
        self.assertNotIn("- basis: 제외조건", prompt)
        self.assertIn("complaints는 정확히 1개인 배열", prompt)

    def test_document_focus_acknowledges_provided_document_names(self) -> None:
        policy = self.complete_policy(제외조건="주택 소유자 제외")
        persona = {**self.persona, "uuid": "c", "age": 25}

        prompt = citizen_prompt(persona, policy)

        self.assertIn("- basis: 구비서류", prompt)
        self.assertIn("제공된 구비서류 이름을 인정한 상태", prompt)
        self.assertIn("서류 항목 자체가 없거나 어떤 서류인지 모른다고 말하지 말 것", prompt)

    def test_missing_exclusion_precedes_safe_provided_field_focus(self) -> None:
        policy = self.complete_policy(제외조건="")
        persona = {**self.persona, "age": 25}

        prompt = citizen_prompt(persona, policy)

        self.assertIn("- 유형: exclusion_gap", prompt)
        self.assertIn("- basis: 제외조건", prompt)
        self.assertIn("별도 제외조건 항목이 비어 있음을 지적", prompt)
        self.assertIn("입력에 없는 자격·소득·재산·주택·중복수급", prompt)

    def test_blank_exclusion_field_is_distinguished_from_selection_text(self) -> None:
        policy = self.complete_policy(
            선정기준="주택 소유자는 지원 대상에서 제외",
            제외조건="",
        )
        persona = {**self.persona, "age": 25}

        focus = _complaint_focus(persona, policy)

        self.assertEqual(focus.kind, "exclusion_gap")
        self.assertIn("다른 정책 항목", focus.policy_evidence)

    def test_unsupported_exclusion_field_is_not_reported_as_missing(self) -> None:
        policy = self.complete_policy()
        policy["상세정보"].pop("제외조건")
        persona = {**self.persona, "age": 25}

        focus = _complaint_focus(persona, policy)

        self.assertNotEqual(focus.kind, "exclusion_gap")
        self.assertNotEqual(focus.basis, "제외조건")

    def test_missing_contact_template_uses_correct_particles(self) -> None:
        policy = self.complete_policy(구비서류="신분증", 문의처="")
        persona = {**self.persona, "age": 25}

        prompt = citizen_prompt(persona, policy)

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

    def test_validation_rejects_same_basis_with_different_focus(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "소득 기준을 더 자세히 설명해 주세요.",
                    "dialogue": "소득 조건을 확인하고 싶습니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            self.persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
            errors,
        )

    def test_validation_accepts_selected_age_focus_evidence(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)

        errors = validate_citizen_response(
            self.valid_result,
            self.persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertFalse(
            any(error.startswith("COMPLAINT_FOCUS_MISMATCH") for error in errors)
        )

    def test_age_focus_must_align_the_dialogue_as_well_as_the_summary(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": (
                        "현재 18세로 19세 이상 기준보다 한 살 낮아 문의합니다."
                    ),
                    "dialogue": "월 20만 원 지원으로는 생활 부담을 줄이기 어렵습니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            self.persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
            errors,
        )

    def test_age_focus_rejects_other_topic_with_both_age_values(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": (
                        "18세와 19세가 온라인으로 신청하기 편한지 궁금합니다."
                    ),
                    "dialogue": "18세와 19세의 신청방법 접근성을 개선해 주세요.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            self.persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
            errors,
        )

    def test_age_focus_rejects_benefit_topic_after_age_evidence(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        claims = (
            "18세와 19세 기준은 알지만 지원이 충분하지 않습니다.",
            "18세와 19세 기준은 알지만 혜택이 부족합니다.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "지원대상",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }
                errors = validate_citizen_response(
                    result,
                    self.persona,
                    self.policy,
                    expected_focus=focus,
                )
                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
                    errors,
                )

    def test_age_focus_rejects_reversed_roles_and_denied_boundary(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        claims = (
            "현재 19세이고 정책은 18세 이상인 연령 기준입니다.",
            "현재 18세가 정책의 19세 이상 연령 기준보다 낮은 것은 아닙니다.",
            (
                "현재 18세이고 정책은 19세 이상이지만 "
                "한 살 차이는 문제가 아닙니다."
            ),
            (
                "현재 기준은 18세이고 저는 19세입니다. "
                "두 연령 기준의 차이와 적용 이유를 설명해 주세요."
            ),
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "지원대상",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }
                errors = validate_citizen_response(
                    result,
                    self.persona,
                    self.policy,
                    expected_focus=focus,
                )
                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
                    errors,
                )

    def test_age_focus_rejects_region_issue_after_valid_age_evidence(self) -> None:
        focus = _complaint_focus(self.persona, self.policy)
        claims = (
            (
                "현재 18세이고 정책은 19세 이상이라 한 살 차이가 있지만, "
                "제 거주지 부산과 정책 지역 서울의 차이를 완화해 주세요."
            ),
            (
                "현재 18세이고 정책은 19세 이상이라 한 살 차이가 있지만, "
                "서울보다 부산도 지원해 주세요."
            ),
            (
                "현재 18세이고 정책은 19세 이상이라 한 살 차이가 있지만, "
                "정책은 서울에만 적용되고 저는 부산에 살아 이 범위를 넓혀야 합니다."
            ),
            (
                "현재 18세이고 정책은 19세 이상이라 한 살 차이가 있지만, "
                "서울 시민만 받고 부산 시민은 못 받으니 넓혀 주세요."
            ),
            (
                "현재 18세이고 정책은 19세 이상이라 한 살 차이가 있지만, "
                "서울 거주자만 받고 부산 거주자는 못 받습니다."
            ),
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "지원대상",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    self.persona,
                    self.policy,
                    expected_focus=focus,
                )

                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
                    errors,
                )

    def test_age_focus_requires_age_units_for_each_boundary_value(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "월 19만 원을 최대 18개월 지원",
            },
        }
        focus = _complaint_focus(self.persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "나이 기준이 궁금합니다.",
                    "dialogue": "월 19만 원을 최대 18개월 지원하는 규모가 적습니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            self.persona,
            policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence",
            errors,
        )

    def test_region_focus_requires_the_mismatched_districts(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "district": "서울-강남구",
        }
        focus = _complaint_focus(persona, self.policy)
        result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
                "거주지": "서울-강남구",
            },
            "grounding": build_grounding_facts(persona, self.policy),
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "서울 정책의 지원 대상 기준이 궁금합니다.",
                    "dialogue": "추가 기준을 설명해 주세요.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence",
            errors,
        )

        result["complaints"] = [
            {
                "basis": "지원대상",
                "complaint_text": "정책 지역은 서울 서초구이고 저는 서울 강남구입니다.",
                "dialogue": (
                    "정책 적용 지역은 서초구이고 제 거주지는 강남구라서 "
                    "지역 경계 기준을 설명해 주세요."
                ),
            }
        ]
        aligned_errors = validate_citizen_response(
            result,
            persona,
            self.policy,
            expected_focus=focus,
        )
        self.assertFalse(
            any(
                error.startswith("COMPLAINT_FOCUS_MISMATCH")
                for error in aligned_errors
            )
        )

        natural_claim = (
            "정책이 서울 서초구에만 적용되고 저는 서울 강남구에 거주해 "
            "지원 대상에서 제외되므로 지역 경계 기준을 설명해 주세요."
        )
        result["complaints"] = [
            {
                "basis": "지원대상",
                "complaint_text": natural_claim,
                "dialogue": natural_claim,
            }
        ]
        natural_errors = validate_citizen_response(
            result,
            persona,
            self.policy,
            expected_focus=focus,
        )
        self.assertFalse(
            any(
                error.startswith("COMPLAINT_FOCUS_MISMATCH")
                or error.startswith("UNSUPPORTED_CURRENT_OUTCOME")
                for error in natural_errors
            )
        )

    def test_region_focus_rejects_other_topic_with_both_regions(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "province": "부산",
            "district": "부산-해운대구",
        }
        focus = _complaint_focus(persona, self.policy)
        result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
                "거주지": "부산-해운대구",
            },
            "grounding": build_grounding_facts(persona, self.policy),
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "서울과 부산의 신청방법 접근성이 궁금합니다.",
                    "dialogue": "부산과 서울에서 필요한 구비서류를 알려 주세요.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence",
            errors,
        )

    def test_region_focus_rejects_benefit_topic_after_region_evidence(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "district": "서울-강남구",
        }
        focus = _complaint_focus(persona, self.policy)
        claim = "서초구와 강남구 경계는 알지만 지원이 충분하지 않습니다."
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": claim,
                    "dialogue": claim,
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            self.policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence",
            errors,
        )

    def test_region_focus_rejects_reversed_roles_denial_and_partial_names(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "province": "부산",
            "district": "부산-서구",
        }
        policy = {
            **self.policy,
            "region_province": "부산",
            "region_district": "부산-강서구",
        }
        focus = _complaint_focus(persona, policy)
        claims = (
            "정책 적용 지역은 부산 서구이고 제 거주지는 부산 강서구라서 지역이 다릅니다.",
            "정책 적용 지역은 부산 강서구이고 제 거주지는 부산 서구지만 다르지 않습니다.",
            "정책 적용 지역과 제 거주지는 모두 부산 강서구라서 지역 경계가 다릅니다.",
            (
                "정책 적용 지역은 부산 강서구이고 제 거주지는 부산 서구지만 "
                "지역 경계는 문제가 아닙니다."
            ),
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "지원대상",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }
                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )
                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence",
                    errors,
                )

    def test_region_focus_rejects_age_issue_after_valid_region_evidence(self) -> None:
        persona = {
            **self.persona,
            "age": 25,
            "province": "부산",
            "district": "부산-해운대구",
        }
        focus = _complaint_focus(persona, self.policy)
        claims = (
            (
                "정책 적용 지역은 서울이고 제 거주지는 부산이라 다르지만, "
                "제 나이 25세에 맞도록 연령 기준을 완화해 주세요."
            ),
            (
                "정책 적용 지역은 서울이고 제 거주지는 부산이라 다르지만, "
                "19세 기준을 완화해 주세요."
            ),
            (
                "정책 적용 지역은 부산이고 제 거주지는 서울이라 "
                "지역 경계 기준의 이유가 궁금합니다."
            ),
            (
                "정책지역과 거주지는 부산과 서울입니다. "
                "두 지역은 서로 다르므로 기준을 설명해 주세요."
            ),
            (
                "정책 적용 지역은 서울이고 제 거주지는 부산이라 다르지만, "
                "19세 이상 34세 이하 구분도 없애 주세요."
            ),
            (
                "정책 적용 지역은 서울이고 제 거주지는 부산이라 다르지만, "
                "청년층만 받고 중장년층은 못 받는 제한도 완화해 주세요."
            ),
            (
                "정책 적용 지역은 서울이고 제 거주지는 부산이라 다르지만, "
                "스무 살 미만은 못 받는 제한도 완화해 주세요."
            ),
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "persona_summary": {
                        **self.valid_result["persona_summary"],
                        "나이": "25",
                        "거주지": "부산-해운대구",
                    },
                    "grounding": build_grounding_facts(persona, self.policy),
                    "complaints": [
                        {
                            "basis": "지원대상",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    self.policy,
                    expected_focus=focus,
                )

                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence",
                    errors,
                )

    def test_gap_focus_requires_field_and_information_gap_evidence(self) -> None:
        persona = {**self.persona, "age": 25}
        cases = (
            self.complete_policy(지원대상=""),
            self.complete_policy(선정기준=""),
            self.complete_policy(제외조건=""),
        )
        for policy in cases:
            focus = _complaint_focus(persona, policy)
            with self.subTest(kind=focus.kind):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": focus.basis,
                            "complaint_text": "해당 기준이 지나치게 엄격합니다.",
                            "dialogue": "기준을 완화해 주세요.",
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertTrue(
                    any(
                        error.startswith(
                            f"COMPLAINT_FOCUS_MISMATCH:{focus.kind}"
                        )
                        for error in errors
                    )
                )

    def test_validation_accepts_exclusion_gap_uncertainty(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": "제외조건이 안내되지 않아 확인이 필요합니다.",
                    "dialogue": (
                        "어떤 경우에 제외되는지 안내가 없어 확인하기 어렵습니다."
                    ),
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertFalse(
            any(error.startswith("COMPLAINT_FOCUS_MISMATCH") for error in errors)
        )

    def test_gap_focus_rejects_claim_that_missing_field_was_published(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": (
                        "제외조건이 상세히 공개되어 있지만 너무 엄격합니다."
                    ),
                    "dialogue": "공개된 제외조건을 완화해 주세요.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:exclusion_gap:missing_gap_evidence",
            errors,
        )

    def test_gap_focus_rejects_strictness_without_missing_information(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": "제외조건이 왜 엄격한지 궁금합니다.",
                    "dialogue": "제외조건을 완화할 수 있는지 궁금합니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:exclusion_gap:missing_gap_evidence",
            errors,
        )

    def test_gap_focus_accepts_explicit_missing_information_inflections(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": (
                        "지원 대상에서 제외되는 기준이 안내되어 있지 않습니다."
                    ),
                    "dialogue": "제외조건을 확인할 수 없어 문의합니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertFalse(
            any(error.startswith("COMPLAINT_FOCUS_MISMATCH") for error in errors)
        )

    def test_gap_focus_accepts_natural_target_inflections(self) -> None:
        persona = {**self.persona, "age": 25}
        cases = (
            (
                self.complete_policy(지원대상=""),
                "누가 혜택을 받을 수 있는지 안내가 없습니다.",
            ),
            (
                self.complete_policy(선정기준=""),
                "누구를 우선 뽑는지 설명이 없습니다.",
            ),
            (
                self.complete_policy(제외조건=""),
                "누가 지원에서 빠지는지 안내가 없습니다.",
            ),
        )
        for policy, text in cases:
            focus = _complaint_focus(persona, policy)
            with self.subTest(kind=focus.kind):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": focus.basis,
                            "complaint_text": text,
                            "dialogue": text,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertFalse(
                    any(
                        error.startswith("COMPLAINT_FOCUS_MISMATCH")
                        for error in errors
                    )
                )

    def test_gap_focus_rejects_negated_missing_information_claims(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        claims = (
            "제외조건이 안내되지 않은 것은 아닙니다.",
            "제외조건 정보가 없는 것은 아닙니다.",
            "제외조건을 알 수 없는 것은 아닙니다.",
            "제외조건 확인이 어렵지는 않습니다.",
            "제외조건이 안내되지 않았다는 설명은 틀렸습니다.",
            "제외조건 정보가 없다는 설명은 틀렸습니다.",
            "제외조건 정보가 없다고 한 것은 사실이 아닙니다.",
            "제외조건 정보가 불분명하지 않습니다.",
            "제외조건 안내가 부족하지 않습니다.",
            "제외조건 정보가 누락되지 않았습니다.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "제외조건",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:exclusion_gap:missing_gap_evidence",
                    errors,
                )

    def test_gap_focus_does_not_merge_evidence_from_different_issues(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": (
                        "제외조건이 너무 엄격합니다. 신청방법 설명이 부족합니다."
                    ),
                    "dialogue": (
                        "제외조건을 완화해 주세요. 신청 절차가 궁금합니다."
                    ),
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:exclusion_gap:missing_gap_evidence",
            errors,
        )

    def test_gap_focus_rejects_additional_issue_and_positive_availability(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(제외조건="")
        focus = _complaint_focus(persona, policy)
        claims = (
            "제외조건 정보가 없습니다. 신청방법 정보도 없습니다.",
            "제외조건 정보가 없습니다. 접수 창구를 알려 주세요.",
            "제외조건 정보가 없습니다. 마감일을 알려 주세요.",
            "제외조건 정보가 없습니다. 전화번호를 알려 주세요.",
            "제외조건 정보가 없습니다. 어디로 신청서를 내야 하는지 알려 주세요.",
            "제외조건 정보가 없습니다. 신청을 언제까지 해야 하는지 알려 주세요.",
            "제외조건 정보가 없습니다. 증빙자료를 알려 주세요.",
            "제외조건 정보가 없습니다. 어느 부서에 전화해야 하는지 알려 주세요.",
            "제외조건은 공개됐지만 제외조건 안내는 없습니다.",
            "제외조건 정보가 없다는 것은 사실이 아닙니다.",
            "제외조건 정보가 없다는 그런 설명 자체는 사실이 아닙니다.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "제외조건",
                            "complaint_text": claim,
                            "dialogue": claim,
                        }
                    ],
                }
                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )
                self.assertIn(
                    "COMPLAINT_FOCUS_MISMATCH:exclusion_gap:missing_gap_evidence",
                    errors,
                )

    def test_gap_focus_allows_positive_context_from_another_field(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(지원대상="")
        focus = _complaint_focus(persona, policy)
        text = (
            "신청방법은 온라인으로 안내되어 있지만, "
            "별도 지원대상 항목은 안내되지 않았습니다."
        )
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": text,
                    "dialogue": text,
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertFalse(
            any(error.startswith("COMPLAINT_FOCUS_MISMATCH") for error in errors)
        )

    def test_gap_focus_rejects_anchor_used_as_owner_or_container(self) -> None:
        persona = {**self.persona, "age": 25}
        cases = (
            (
                self.complete_policy(지원대상=""),
                "지원대상자의 의견은 안내되지 않았습니다.",
            ),
            (
                self.complete_policy(지원대상=""),
                "지원대상에 대한 제 의견은 안내되지 않았습니다.",
            ),
            (
                self.complete_policy(지원대상=""),
                "접수 안내는 지원대상자에게 제공되지 않았습니다.",
            ),
            (
                self.complete_policy(제외조건=""),
                "지원대상 정보는 제외조건 항목에 안내되지 않았습니다.",
            ),
            (
                self.complete_policy(신청방법=""),
                "신청방법에 대한 제 의견은 안내되지 않았습니다.",
            ),
            (
                self.complete_policy(신청방법=""),
                "담당 부서 연결 정보가 신청방법에 안내되지 않았습니다.",
            ),
        )
        for policy, text in cases:
            focus = _complaint_focus(persona, policy)
            with self.subTest(kind=focus.kind, text=text):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": focus.basis,
                            "complaint_text": text,
                            "dialogue": text,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertTrue(
                    any(error.startswith("COMPLAINT_FOCUS_MISMATCH") for error in errors)
                )

    def test_missing_admin_focus_rejects_a_different_missing_field(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy(신청방법="")
        focus = _complaint_focus(persona, policy)
        self.assertEqual(focus.kind, "missing_신청방법")
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "정보미제공",
                    "complaint_text": "신청기한이 안내되지 않아 확인하기 어렵습니다.",
                    "dialogue": "접수 기간 정보가 없어 확인할 수 없습니다.",
                }
            ],
        }

        errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )

        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:missing_신청방법:missing_gap_evidence",
            errors,
        )

        result["complaints"] = [
            {
                "basis": "정보미제공",
                "complaint_text": (
                    "신청방법 정보가 없습니다. 신청기한 정보도 없습니다."
                ),
                "dialogue": (
                    "신청방법 안내가 없습니다. 구비서류 정보도 없습니다."
                ),
            }
        ]
        mixed_errors = validate_citizen_response(
            result,
            persona,
            policy,
            expected_focus=focus,
        )
        self.assertIn(
            "COMPLAINT_FOCUS_MISMATCH:missing_신청방법:missing_gap_evidence",
            mixed_errors,
        )

    def test_safe_focuses_reject_an_unrelated_policy_name_request(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy()
        cases = (
            ("benefit_effectiveness", "지원내용"),
            ("application_accessibility", "신청방법"),
            ("deadline_burden", "신청기한"),
            ("document_burden", "구비서류"),
        )
        for kind, basis in cases:
            with self.subTest(kind=kind):
                focus = ComplaintFocus(
                    kind=kind,
                    basis=basis,
                    priority=10,
                    policy_evidence="테스트 근거",
                    persona_evidence="테스트 근거",
                    question_seed="테스트 질문",
                    instruction="테스트 지시",
                )
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": basis,
                            "complaint_text": "정책 이름을 더 쉽게 바꿔 주세요.",
                            "dialogue": "정책 이름을 짧게 고쳐 주세요.",
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertIn(
                    f"COMPLAINT_FOCUS_MISMATCH:{kind}:missing_topic_evidence",
                    errors,
                )

    def test_safe_focuses_accept_their_own_topic_and_concern(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy()
        cases = (
            (
                "benefit_effectiveness",
                "지원내용",
                "지원금 규모가 생활 부담을 줄이기에 충분한지 궁금합니다.",
            ),
            (
                "application_accessibility",
                "신청방법",
                "신청방법의 접근성과 이용 편의가 충분한지 궁금합니다.",
            ),
            (
                "deadline_burden",
                "신청기한",
                "신청기한이 준비하기에 충분한지 궁금합니다.",
            ),
            (
                "document_burden",
                "구비서류",
                "구비서류의 준비와 제출 절차 부담을 안내해 주세요.",
            ),
        )
        for kind, basis, text in cases:
            with self.subTest(kind=kind):
                focus = ComplaintFocus(
                    kind=kind,
                    basis=basis,
                    priority=10,
                    policy_evidence="테스트 근거",
                    persona_evidence="테스트 근거",
                    question_seed="테스트 질문",
                    instruction="테스트 지시",
                )
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": basis,
                            "complaint_text": text,
                            "dialogue": text,
                        }
                    ],
                }

                errors = validate_citizen_response(
                    result,
                    persona,
                    policy,
                    expected_focus=focus,
                )

                self.assertFalse(
                    any(
                        error.startswith("COMPLAINT_FOCUS_MISMATCH")
                        for error in errors
                    )
                )

    def test_safe_focuses_require_a_real_single_issue_concern(self) -> None:
        persona = {**self.persona, "age": 25}
        policy = self.complete_policy()
        cases = (
            (
                "benefit_effectiveness",
                "지원내용",
                (
                    "지원 규모는 월 20만 원입니다.",
                    "지원 내용은 충분하고 효과도 좋아 아무런 문제가 없습니다.",
                    "지원금 규모는 충분하지만 신청방법 접근성이 어렵습니다.",
                    "지원금이 충분한지는 분명합니다.",
                    "지원 규모가 충분한지 확인했고 아주 충분합니다.",
                    "지원 규모가 충분한지 살펴봤으며 현실적으로 아주 충분합니다.",
                    "지원 규모는 월 20만 원이고 담당자 태도가 걱정입니다.",
                    "지원내용이 충분한지 여부와 신청방법 불편 개선이 필요합니다.",
                ),
            ),
            (
                "application_accessibility",
                "신청방법",
                (
                    "신청 절차는 온라인입니다.",
                    "온라인 신청방법은 쉽고 접근성이 좋아 아무런 문제가 없습니다.",
                    "신청방법 접근성은 좋지만 지원금 규모가 부족합니다.",
                    "신청방법이 편리한지는 분명합니다.",
                    "신청방법이 편리한지 확인했고 매우 편리합니다.",
                    "신청방법 접근성이 편리한지 살펴보면 실제로 편리합니다.",
                    "신청방법은 온라인이고 담당자 태도가 걱정입니다.",
                ),
            ),
            (
                "deadline_burden",
                "신청기한",
                (
                    "신청기간은 상시입니다.",
                    "신청기한은 여유 있고 충분해서 준비 부담이 전혀 없습니다.",
                    "신청기한은 충분하지만 구비서류 준비가 부담됩니다.",
                    "신청기한이 충분한지는 분명합니다.",
                    "신청기간이 충분한지 확인했고 아주 충분합니다.",
                    "신청기한은 촉박하지 않고 준비 부담이 없습니다.",
                    "신청기한은 상시이고 담당자 태도가 걱정입니다.",
                ),
            ),
            (
                "document_burden",
                "구비서류",
                (
                    "제출서류는 신분증과 주민등록등본입니다.",
                    "구비서류는 준비가 쉽고 제출 부담이 전혀 없습니다.",
                    "구비서류 준비는 쉽지만 신청기한이 촉박합니다.",
                    "서류 준비가 어렵지 않다는 점은 분명합니다.",
                    "구비서류는 신분증이고 담당자 태도가 걱정입니다.",
                ),
            ),
        )
        for kind, basis, claims in cases:
            focus = ComplaintFocus(
                kind=kind,
                basis=basis,
                priority=10,
                policy_evidence="테스트 근거",
                persona_evidence="테스트 근거",
                question_seed="테스트 질문",
                instruction="테스트 지시",
            )
            for claim in claims:
                with self.subTest(kind=kind, claim=claim):
                    result = {
                        **self.valid_result,
                        "complaints": [
                            {
                                "basis": basis,
                                "complaint_text": claim,
                                "dialogue": claim,
                            }
                        ],
                    }
                    errors = validate_citizen_response(
                        result,
                        persona,
                        policy,
                        expected_focus=focus,
                    )
                    self.assertTrue(
                        any(
                            error.startswith(
                                f"COMPLAINT_FOCUS_MISMATCH:{kind}"
                            )
                            for error in errors
                        )
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
        policy = self.complete_policy(
            지원내용="월 최대 20만 원을 최대 12개월 지원"
        )
        persona = {**self.persona, "age": 25}
        generated = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(persona, policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "제공된 지원 규모가 충분한지 의문입니다.",
                    "dialogue": "월 20만 원을 최대 12개월 지원하는 규모가 충분한가요?",
                }
            ],
        }
        run_llm.return_value = json.dumps(generated, ensure_ascii=False)

        result = run_citizen_simulation(persona, policy, max_retries=3)

        self.assertIsNotNone(result)
        self.assertEqual(run_llm.call_count, 1)
        self.assertIn("월 최대 20만 원", result["complaints"][0]["dialogue"])
        self.assertEqual(result["_quality_gate"]["generation_attempts"], 1)

    @patch(
        "backend.ai_simulation_core.simulations.citizen_simulation.run_llm"
    )
    def test_semantic_failure_feedback_is_used_on_retry(self, run_llm) -> None:
        eligible_persona = {**self.persona, "age": 25}
        policy = self.complete_policy()
        valid = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, policy),
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
            policy,
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
        policy = self.complete_policy()
        mixed_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, policy),
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
            policy,
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
        policy = self.complete_policy()
        valid_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, policy),
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
            policy,
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
        policy = self.complete_policy()
        valid_result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "나이": "25",
            },
            "grounding": build_grounding_facts(eligible_persona, policy),
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
            policy,
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
