import unittest

from backend.ai_simulation_core.simulations.citizen_quality import (
    MATCHED,
    NOT_MATCHED,
    build_grounding_facts,
    evaluate_region_status,
    validate_semantic_quality,
)
from backend.ai_simulation_core.region_matching import region_matches


class CitizenQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.persona = {
            "uuid": "eligible-excel",
            "age": 25,
            "occupation": "사무 보조원",
            "sex": "여자",
            "province": "서울",
            "district": "서울-관악구",
            "persona": "서울 관악구에 사는 꼼꼼한 사무 보조원입니다.",
            "professional_persona": "엑셀 수식을 완벽하게 정리하는 실무자입니다.",
            "family_persona": "관악구의 자가 주택에서 독립해 생활합니다.",
        }
        self.policy = {
            "region_scope": "specific",
            "region_province": "서울",
            "region_district": "",
            "age_min": 19,
            "age_max": 34,
            "age_basis": "dataset_age",
            "상세정보": {
                "서비스명": "서울 청년 주거비 지원",
                "지원대상": "서울 거주 만 19세 이상 34세 이하 청년",
                "선정기준": "",
                "지원내용": "월 최대 20만 원 지원",
                "신청방법": "",
                "신청기한": "",
                "구비서류": "",
                "제외조건": "",
                "문의처": "",
            },
        }
        self.valid_result = {
            "persona_summary": {
                "이름": "김민지",
                "직업": "사무 보조원",
                "성별": "여자",
                "나이": "25",
                "거주지": "서울-관악구",
            },
            "grounding": build_grounding_facts(self.persona, self.policy),
            "personality": "정책 안내를 꼼꼼히 확인하는 사무 보조원입니다.",
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 최대 20만 원의 지원 규모가 충분한지 걱정됩니다.",
                    "dialogue": "월 최대 20만 원으로 주거 부담을 얼마나 덜 수 있는지 궁금해요.",
                }
            ],
        }

    def errors_for(self, **changes) -> list[str]:
        result = {**self.valid_result, **changes}
        return validate_semantic_quality(result, self.persona, self.policy)

    def test_grounded_result_passes(self) -> None:
        self.assertEqual(
            validate_semantic_quality(
                self.valid_result,
                self.persona,
                self.policy,
            ),
            [],
        )

    def test_quality_and_candidate_region_matching_use_the_same_matrix(self) -> None:
        cases = (
            ({"province": "서울", "district": "서울-서초구"}, "서울", "서초구"),
            ({"province": "서울", "district": "서초구"}, "서울", "서울-서초구"),
            ({"province": "서울", "district": "서울-강남구"}, "서울", "서초구"),
            ({"province": "서울", "district": "서울-강서구"}, "서울", "서구"),
            ({"province": "부산", "district": "부산-중구"}, "서울", "서울-중구"),
            ({"province": "서울", "district": ""}, "서울", "서초구"),
        )
        for persona, province, district in cases:
            with self.subTest(persona=persona, district=district):
                matches = region_matches(
                    persona,
                    region_scope="specific",
                    province=province,
                    district=district,
                )
                status = evaluate_region_status(
                    persona,
                    {
                        "region_scope": "specific",
                        "region_province": province,
                        "region_district": district,
                    },
                )
                self.assertEqual(status, MATCHED if matches else NOT_MATCHED)

    def test_eligible_persona_rejects_age_exclusion_claim(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원대상",
                    "complaint_text": "나는 25세인데 나이 때문에 지원 대상이 되지 않습니다.",
                    "dialogue": "25세가 되면 지원을 못 받는다니 답답해요.",
                }
            ]
        )
        self.assertTrue(any(error.startswith("AGE_ELIGIBILITY") for error in errors))

    def test_eligible_persona_rejects_region_exclusion_claim(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원대상",
                    "complaint_text": "서울 거주지 조건 때문에 지원 대상에서 제외됩니다.",
                    "dialogue": "서울에 산다는 지역 조건 때문에 지원을 못 받는 건 부당해요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("REGION_ELIGIBILITY") for error in errors)
        )

    def test_policy_name_does_not_create_age_or_region_exclusion_cause(self) -> None:
        for service_name in ("서울 청년 주거비 지원", "지역 청년 주거비 지원"):
            with self.subTest(service_name=service_name):
                policy = {
                    **self.policy,
                    "상세정보": {
                        **self.policy["상세정보"],
                        "서비스명": service_name,
                    },
                }
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "제외조건",
                            "complaint_text": (
                                f"제외조건이 안내되지 않아 어떤 경우에 {service_name} "
                                "대상에서 제외되는지 확인이 필요합니다."
                            ),
                            "dialogue": (
                                f"{service_name}의 제외조건이 명시되지 않아 어떤 경우에 "
                                "지원 대상에서 제외되는지 궁금합니다."
                            ),
                        }
                    ],
                }

                errors = validate_semantic_quality(result, self.persona, policy)

                self.assertFalse(
                    any(error.startswith("AGE_ELIGIBILITY") for error in errors)
                )
                self.assertFalse(
                    any(error.startswith("REGION_ELIGIBILITY") for error in errors)
                )

    def test_explicit_age_exclusion_causes_are_rejected(self) -> None:
        sentences = (
            "저는 청년이 아니라 지원 대상이 아니라고 합니다.",
            "청년 기준 때문에 지원을 못 받습니다.",
            "청년이라서 지원 대상에서 제외됩니다.",
            "연령 요건을 충족하지 못해 지원 대상에서 제외됩니다.",
            "연령 제한에 걸려 지원 대상에서 탈락했습니다.",
            "나이가 맞지 않아 지원을 받을 수 없습니다.",
            "청년 기준에서 벗어나 지원 대상이 되지 않습니다.",
            "나이 때문에 지원 대상이 아닙니다.",
            "나이 때문에 신청해도 지원 대상에서 제외됩니다.",
            "나이 때문에 청년 주택 지원 대상에서 제외됩니다.",
            "연령 기준 때문에 추가 지원 대상에서 제외됩니다.",
            "연령 초과로 지원 대상에서 제외됩니다.",
            "나이 상한을 초과해 지원 대상에서 제외됩니다.",
            "나이가 기준에 미달해 지원 대상에서 제외됩니다.",
            "나이 때문에 신청 자격이 없어 지원 대상에서 제외됩니다.",
            "연령 때문에 접수조차 못 하고 지원 대상에서 제외됩니다.",
            "지원 대상이 아닌 이유는 나이 기준 때문입니다.",
            "나이 기준에는 맞지만 결국 나이 때문에 지원 대상에서 제외됩니다.",
            "나이를 이유로 지원을 받지 못합니다.",
            "연령 탓에 지원 대상에서 제외됩니다.",
            "나이가 어려서 지원 대상에서 제외됩니다.",
            "청년층이라는 이유로 지원 대상에서 제외됩니다.",
            "25세라는 사실만으로 별도의 추가 심사 없이 지원 대상에서 제외됩니다.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "지원대상",
                            "complaint_text": sentence,
                            "dialogue": sentence,
                        }
                    ]
                )

                self.assertTrue(
                    any(error.startswith("AGE_ELIGIBILITY") for error in errors)
                )

    def test_explicit_region_exclusion_causes_are_rejected(self) -> None:
        sentences = (
            "서울 시민이라 지원 대상이 아니라고 합니다.",
            "거주지가 정책에서 정한 범위와 일치하지 않는 조건 때문에 지원 대상에서 제외됩니다.",
            "거주지 조건 때문에 대상자가 아닙니다.",
            "거주지 조건 때문에 주택 지원 대상에서 제외됩니다.",
            "서울 시민이라서 청년 주택 지원 대상에서 제외됩니다.",
            "서울 거주 조건에 맞지 않아 지원 대상에서 제외됩니다.",
            "서울 거주 요건을 충족하지 못해 지원 대상에서 제외됩니다.",
            "서울에 살지 않아 지원 대상에서 제외됩니다.",
            "주소지가 서울이 아니어서 지원 대상에서 제외됩니다.",
            "거주지가 서울과 일치하지 않아 지원 대상에서 제외됩니다.",
            "거주지 요건에서 벗어나 지원 대상에서 제외됩니다.",
            "지역 제한에 걸려 지원 대상에서 제외됩니다.",
            "거주지 조건으로 인해 지원 대상에서 제외됩니다.",
            "추가 거주지 제한 때문에 지원 대상에서 제외됩니다.",
            "지원 대상이 아닌 이유는 거주지 기준 때문입니다.",
            "지역 기준에는 맞지만 결국 거주지 때문에 지원 대상에서 제외됩니다.",
            "서울 관할이 아니라서 지원 대상에서 제외됩니다.",
            "서울 소속이 아니어서 지원 대상에서 제외됩니다.",
            "서울 관할 조건을 충족하지 못해 지원 대상에서 제외됩니다.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "지원대상",
                            "complaint_text": sentence,
                            "dialogue": sentence,
                        }
                    ]
                )

                self.assertTrue(
                    any(error.startswith("REGION_ELIGIBILITY") for error in errors)
                )

    def test_other_causes_do_not_become_age_or_region_contradictions(self) -> None:
        sentences = (
            "서울 청년 지원이라서 어떤 경우에 지원 대상에서 제외되는지 궁금합니다.",
            "연령 기준상 대상이지만 재산 때문에 지원을 받지 못합니다.",
            "25세라 신청했지만 소득 때문에 제외됐습니다.",
            "서울이라서 신청했지만 소득 때문에 지원 대상에서 제외됩니다.",
            "서울에서 거주하기 때문에 신청했고 서류 문제로 지원 대상에서 제외됐습니다.",
            "서울에 거주하며 지역 요건도 충족하지만 소득 때문에 지원 대상에서 제외됩니다.",
            "나이 때문이 아니라 소득 때문에 지원 대상에서 제외됩니다.",
            "서울 거주지 때문이 아니라 재산 때문에 지원 대상에서 제외됩니다.",
            "연령 조건을 충족하지 못한 것이 아니라 소득 때문에 지원 대상에서 제외됩니다.",
            "거주지 조건을 충족하지 못한 것이 아니라 재산 때문에 지원 대상에서 제외됩니다.",
            "나이가 맞지 않을까 걱정했지만 실제 연령은 충족하고 소득 때문에 지원 대상에서 제외됩니다.",
            "나이 기준이 맞지 않는 게 아니라 취업 상태 때문에 지원 대상에서 제외됩니다.",
            "지역 기준이 맞지 않는 게 아니라 취업 상태 때문에 지원 대상에서 제외됩니다.",
            "나이가 다르지 않고 소득 때문에 지원 대상에서 제외됩니다.",
            "지역이 다르지 않고 소득 때문에 지원 대상에서 제외됩니다.",
            "나이 조건에 맞지 않는 것은 아니지만 소득 때문에 대상자가 아닙니다.",
            "연령 요건을 충족하지 못했다는 설명은 틀렸고 재산 때문에 대상자가 아닙니다.",
            "거주지 조건에 맞지 않는 것은 아니지만 소득 때문에 대상자가 아닙니다.",
            "서울 주민이 아니라는 것은 사실이 아니지만 서류 때문에 대상자가 아닙니다.",
            "나이가 다르지는 않고 지원 대상에서 제외된 이유는 소득입니다.",
            "나이 기준에서 벗어나지는 않았고 지원 대상에서 제외된 이유는 재산입니다.",
            "나이가 기준에 미달하지는 않았고 지원 대상에서 제외된 이유는 소득입니다.",
            "나이가 상한을 초과하지는 않았고 지원 대상에서 제외된 이유는 재산입니다.",
            "지역이 다르지는 않고 지원 대상에서 제외된 이유는 소득입니다.",
            "거주지 범위를 벗어나지는 않았고 지원 대상에서 제외된 이유는 재산입니다.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "제외조건",
                            "complaint_text": sentence,
                            "dialogue": sentence,
                        }
                    ]
                )

                self.assertFalse(
                    any(error.startswith("AGE_ELIGIBILITY") for error in errors)
                )
                self.assertFalse(
                    any(error.startswith("REGION_ELIGIBILITY") for error in errors)
                )

    def test_blank_method_rejects_online_only_claim(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "신청방법",
                    "complaint_text": "온라인으로만 신청해야 하고 방문은 불가능합니다.",
                    "dialogue": "온라인만 가능하고 주민센터 방문은 안 된다니 불편해요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_POLICY_FACT:신청방법") for error in errors)
        )

    def test_blank_method_allows_information_gap_question(self) -> None:
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "정보미제공",
                    "complaint_text": "신청방법 안내가 없어 확인이 필요합니다.",
                    "dialogue": "온라인인지 방문인지 알 수 없어 어디에 물어봐야 할지 모르겠어요.",
                }
            ],
        }
        self.assertEqual(validate_semantic_quality(result, self.persona, self.policy), [])

    def test_persona_use_of_today_is_not_a_deadline_claim(self) -> None:
        result = {
            **self.valid_result,
            "personality": "상사의 오늘 기분을 세심하게 읽고 업무를 조절합니다.",
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, self.policy), [])

    def test_provided_target_cannot_be_claimed_as_unknown(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원대상",
                    "complaint_text": "지원 대상이 무엇인지 정확히 알 수 없습니다.",
                    "dialogue": "지원 대상 안내가 전혀 없어서 확인이 필요합니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:지원대상") for error in errors)
        )

    def test_provided_service_name_cannot_be_claimed_as_unknown(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "정보미제공",
                    "complaint_text": "서비스명과 소관기관명이 정보 없음으로 기재되어 있습니다.",
                    "dialogue": "정책명이 전혀 안내되지 않아 어떤 사업인지 모르겠습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:서비스명") for error in errors)
        )

    def test_provided_fields_allow_missing_detail_criticism(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청방법": "온라인 신청 또는 주민센터 방문",
                "문의처": "청년지원과 02-1234-5678",
                "제외조건": "주택 소유자 제외",
            },
        }
        complaints = [
            {
                "basis": "문의처",
                "complaint_text": "문의처가 있지만 운영 시간이 없어 확인이 필요합니다.",
                "dialogue": "전화 상담 가능 시간이 안내되면 좋겠습니다.",
            },
            {
                "basis": "신청방법",
                "complaint_text": "신청 방법은 안내되어 있지만 접근성 지원 설명이 없습니다.",
                "dialogue": "장애인 접근성 지원이 있는지 궁금합니다.",
            },
            {
                "basis": "제외조건",
                "complaint_text": "제외 조건은 있지만 공동명의 기준이 없어 확인이 필요합니다.",
                "dialogue": "공동명의 주택의 적용 기준을 알고 싶습니다.",
            },
        ]
        result = {**self.valid_result, "complaints": complaints}

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_provided_documents_reject_claim_that_document_list_is_unknown(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "Identification card and proof of residence",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "신청서에 필요한 서류가 명확히 안내되지 않았습니다.",
                    "dialogue": "어떤 서류를 준비해야 할지 전혀 안내가 없어서 모르겠습니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)
        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:구비서류") for error in errors)
        )

    def test_provided_documents_allow_missing_preparation_detail_criticism(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증과 거주 증명서",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "신분증과 거주 증명서는 안내되어 있지만 발급처가 없습니다.",
                    "dialogue": "두 서류의 발급 방법과 제출 방식이 안내되면 좋겠습니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_provided_documents_reject_invented_visit_requirement(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "신분증, 거주 증명서",
                "신청방법": "온라인 또는 주민센터 방문 신청",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "거주 증명서를 발급받기 위해 방문해야 합니다.",
                    "dialogue": "서류 발급을 위한 방문 절차가 부담스럽습니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)
        self.assertTrue(
            any(
                error.startswith("UNSUPPORTED_DOCUMENT_PROCEDURE")
                for error in errors
            )
        )

    def test_provided_documents_allow_channel_questions(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "Identification card and proof of residence",
                "신청방법": "온라인 또는 주민센터 방문 신청",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": (
                        "신분증과 거주 증명서의 준비 방법에 구체적인 안내가 필요합니다."
                    ),
                    "dialogue": (
                        "신분증과 거주 증명서를 어디서 발급받고, 방문이나 온라인 중 "
                        "어떻게 제출하는지 궁금합니다."
                    ),
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_provided_documents_require_all_recognized_names(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "Identification card and proof of residence",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "신청서의 발급처와 제출 방법 안내가 부족합니다.",
                    "dialogue": "어떤 경로로 준비하는지 확인하고 싶습니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)
        self.assertTrue(
            any(error.startswith("DOCUMENT_IDENTITY_MISSING") for error in errors)
        )

    def test_provided_benefit_cannot_be_grouped_with_missing_fields(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "정보미제공",
                    "complaint_text": "지원 내용이나 신청 방법, 기한 등 어떤 정보도 없습니다.",
                    "dialogue": "지원 내용이 전혀 안내되지 않아 알 수 없습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:지원내용") for error in errors)
        )

    def test_provided_support_amount_cannot_be_claimed_as_unknown(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원내용",
                    "complaint_text": "월 최대 20만 원 지원입니다.",
                    "dialogue": "지원금이 얼마나 지급되는지 전혀 모르겠어요.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:지원내용") for error in errors)
        )

    def test_provided_support_amount_can_be_criticized_as_insufficient(self) -> None:
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 최대 20만 원이 충분한지 모르겠습니다.",
                    "dialogue": "지원 규모가 현실적인 부담을 덜기에 부족할 수 있습니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, self.policy), [])

    def test_actual_application_history_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "지원을 신청했지만 실제로 탈락했습니다.",
                    "dialogue": "신청을 했는데 거절당해서 억울해요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_APPLICATION_HISTORY") for error in errors)
        )

    def test_invented_negative_support_history_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "제외조건",
                    "complaint_text": "중복 지원 여부를 확인해 주세요.",
                    "dialogue": "저는 아직 다른 주거비 지원을 받지 않았습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_APPLICATION_HISTORY") for error in errors)
        )

    def test_current_denial_presupposition_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원대상",
                    "complaint_text": "나이와 지역 조건은 충족합니다.",
                    "dialogue": "저는 조건에 맞는데 왜 지원이 안 되는지 모르겠어요.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_CURRENT_OUTCOME") for error in errors)
        )

    def test_structured_region_mismatch_may_explain_current_exclusion(self) -> None:
        persona = {
            **self.persona,
            "province": "부산",
            "district": "부산-해운대구",
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(persona, self.policy),
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": (
                        "정책 지역은 서울이고 현재 거주지는 부산이라 "
                        "지원 대상에서 제외됩니다."
                    ),
                    "dialogue": (
                        "정책이 서울에만 적용되어 부산 거주자는 "
                        "지원 대상에서 제외됩니다."
                    ),
                }
            ],
        }

        errors = validate_semantic_quality(result, persona, self.policy)

        self.assertFalse(
            any(error.startswith("UNSUPPORTED_CURRENT_OUTCOME") for error in errors)
        )

    def test_conditional_denial_concern_is_allowed(self) -> None:
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "선정기준",
                    "complaint_text": "다른 조건 때문에 지원이 안 될까 걱정됩니다.",
                    "dialogue": "추가 선정 기준이 있는지 확인이 필요합니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, self.policy), [])

    def test_overall_eligibility_assertion_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원대상",
                    "complaint_text": "저는 지원 대상이 맞습니다.",
                    "dialogue": "자격이 있어서 지원받을 수 있습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_OVERALL_ELIGIBILITY") for error in errors)
        )

    def test_concessive_overall_eligibility_assertion_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "신청방법",
                    "complaint_text": "신청방법의 세부 안내가 더 필요합니다.",
                    "dialogue": "온라인 신청으로 지원을 받을 수 있지만 절차가 복잡합니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_OVERALL_ELIGIBILITY") for error in errors)
        )

    def test_long_overall_eligibility_assertion_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "구비서류",
                    "complaint_text": "신분증과 거주 증명서 안내가 더 필요합니다.",
                    "dialogue": (
                        "저는 유치원 교사인데 서울 청년 주거 지원 정책에 "
                        "자격이 있다고 생각합니다."
                    ),
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_OVERALL_ELIGIBILITY") for error in errors)
        )

    def test_conditional_overall_eligibility_question_is_allowed(self) -> None:
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "지역과 나이 외 다른 조건은 확인이 필요합니다.",
                    "dialogue": "전체 지원 자격에 추가 기준이 있는지 궁금합니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, self.policy), [])

    def test_positive_current_outcome_assertion_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "저는 선정되었습니다.",
                    "dialogue": "지원금을 지급받고 있습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_CURRENT_OUTCOME") for error in errors)
        )

    def test_subjective_personal_financial_concern_is_allowed(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "저는 생활비 부담이 너무 큽니다.",
                    "dialogue": "제가 경제적으로 어려워서 지원이 필요합니다.",
                }
            ]
        )

        self.assertEqual(errors, [])

    def test_invented_family_medical_and_childcare_expenses_are_rejected(self) -> None:
        for dialogue in (
            "월 최대 20만 원으로는 아내의 병원비를 마련하기에 부족합니다.",
            "월 최대 20만 원으로 아이를 키우는 생활비를 감당하기 어렵습니다.",
        ):
            with self.subTest(dialogue=dialogue):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "지원내용",
                            "complaint_text": "지원 규모의 실효성이 궁금합니다.",
                            "dialogue": dialogue,
                        }
                    ]
                )
                self.assertTrue(
                    any(
                        error.startswith("UNSUPPORTED_FAMILY_FINANCIAL_FACT")
                        for error in errors
                    )
                )

    def test_source_grounded_family_medical_expense_is_allowed(self) -> None:
        persona = {
            **self.persona,
            "family_persona": "배우자의 병원비 부담이 커서 생활이 빠듯합니다.",
        }
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 최대 20만 원 지원의 실효성이 궁금합니다.",
                    "dialogue": "배우자의 병원비 부담을 덜기에는 부족할 수 있습니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, persona, self.policy), [])

    def test_current_personal_preparation_state_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "신청기한",
                    "complaint_text": "현재까지 준비를 하지 못해 기회를 놓칠 상황입니다.",
                    "dialogue": "지금까지 신청 준비를 못 해서 걱정입니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_APPLICATION_HISTORY") for error in errors)
        )

    def test_preparation_start_state_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "신청기한",
                    "complaint_text": (
                        "현재까지 준비를 시작하지 못한 상황에서 시간이 부족합니다."
                    ),
                    "dialogue": "지금까지 준비도 시작하지 못해 걱정입니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("UNSUPPORTED_APPLICATION_HISTORY") for error in errors)
        )

    def test_generic_personal_expense_claims_are_allowed(self) -> None:
        for dialogue in (
            "월 최대 20만 원이 중소기업 전산팀 직원의 생활비를 충당하기에는 부족합니다.",
            "전화 상담원으로 일하고 있어 생활비도 꾸준히 들어가야 합니다.",
            "월 최대 20만 원으로 생활비와 주거비를 충당하기 어렵습니다.",
        ):
            with self.subTest(dialogue=dialogue):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "지원내용",
                            "complaint_text": "지원 규모의 실효성이 궁금합니다.",
                            "dialogue": dialogue,
                        }
                    ]
                )
                self.assertEqual(errors, [])

    def test_subjective_personal_schedule_is_allowed(self) -> None:
        for dialogue in (
            "퇴근 후에도 자격증 공부를 해야 해서 준비 시간이 부족합니다.",
            "일상과 가정 생활을 관리해야 해서 준비가 어렵습니다.",
            "교실에서 아이들을 돌보는 데 시간이 많이 들어 준비가 어렵습니다.",
        ):
            with self.subTest(dialogue=dialogue):
                errors = self.errors_for(
                    complaints=[
                        {
                            "basis": "신청기한",
                            "complaint_text": "제공된 신청기간이 충분한지 의문입니다.",
                            "dialogue": dialogue,
                        }
                    ]
                )
                self.assertEqual(errors, [])

    def test_maximum_support_qualifier_cannot_be_dropped(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원의 지원 규모가 부족합니다.",
                    "dialogue": "월 20만 원 지원으로 부담을 덜기 어렵습니다.",
                }
            ]
        )

        self.assertTrue(
            any(error.startswith("POLICY_QUALIFIER_MISSING") for error in errors)
        )

    def test_later_duration_qualifier_does_not_cover_amount(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "월 최대 20만 원을 최대 12개월 지원",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 최대 20만 원을 최대 12개월 지원합니다.",
                    "dialogue": "월 20만 원을 최대 12개월 지원하는 규모가 충분한가요?",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)

        self.assertTrue(
            any(
                error.startswith(
                    "POLICY_QUALIFIER_MISSING:지원내용:최대:complaints[0].dialogue"
                )
                for error in errors
            )
        )

    def test_late_deadline_cannot_be_claimed_to_reduce_preparation_time(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청기한": "2026-09-30",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "신청기한",
                    "complaint_text": "신청기한을 검토해 주세요.",
                    "dialogue": "신청 기한이 늦어서 준비가 안 되겠어요.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)

        self.assertTrue(
            any(error.startswith("DEADLINE_CAUSALITY_CONTRADICTION") for error in errors)
        )

    def test_full_calendar_month_allows_equivalent_duration(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청기한": "2026-09-01 ~ 2026-09-30",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "신청기한",
                    "complaint_text": "제공된 신청기간이 충분한지 의문입니다.",
                    "dialogue": "신청 기간이 1개월이라 준비하기에 충분한지 궁금합니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_document_question_cannot_deny_provided_application_channel(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "Identification card and proof of residence",
                "신청방법": "Apply online or visit local community service center",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "신분증과 거주 증명서의 세부 절차가 궁금합니다.",
                    "dialogue": "신분증과 거주 증명서를 온라인으로 제출할 수 있는지 모르겠습니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)

        self.assertTrue(
            any(error.startswith("CONTRADICTED_POLICY_FACT:신청방법") for error in errors)
        )

    def test_document_question_may_request_details_with_known_channels(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "구비서류": "Identification card and proof of residence",
                "신청방법": "Apply online or visit local community service center",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "구비서류",
                    "complaint_text": "신분증과 거주 증명서의 세부 절차가 궁금합니다.",
                    "dialogue": "온라인 제출 순서와 방문 센터의 구체적인 위치 안내가 필요합니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_source_persona_name_must_match_summary(self) -> None:
        persona = {
            **self.persona,
            "professional_persona": "이혜진 씨는 꼼꼼한 상담원입니다.",
        }
        result = {
            **self.valid_result,
            "persona_summary": {
                **self.valid_result["persona_summary"],
                "이름": "이지은",
            },
        }

        errors = validate_semantic_quality(result, persona, self.policy)

        self.assertIn("PERSONA_SUMMARY_MISMATCH:이름", errors)

    def test_unseen_money_amount_is_rejected(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "제가 내는 월세는 28만 원입니다.",
                    "dialogue": "내 월세가 28만 원이라서 지원이 더 필요해요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_NUMERIC_FACT") for error in errors)
        )

    def test_krw_amount_allows_equivalent_korean_expression(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "KRW 200000 per month for up to 12 months",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 20만 원 지원으로는 부담을 덜기 어렵습니다.",
                    "dialogue": "최대 12개월 동안 월 20만 원만 지원되어 아쉽습니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_krw_amount_rejects_different_korean_expression(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "지원내용": "KRW 200000 per month for up to 12 months",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "지원내용",
                    "complaint_text": "월 25만 원 지원으로는 부담을 덜기 어렵습니다.",
                    "dialogue": "월 25만 원만 지원되어 아쉽습니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_NUMERIC_FACT") for error in errors)
        )

    def test_iso_policy_dates_allow_equivalent_korean_date_expression(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청기한": "2026-09-01 ~ 2026-09-30",
                "시행일": "2026-10-01",
            },
        }
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "신청기한",
                    "complaint_text": "신청은 2026년 9월 30일까지라 준비 기간이 짧습니다.",
                    "dialogue": "9월 30일까지 서류를 준비해야 하는 점이 부담됩니다.",
                }
            ],
        }

        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_iso_policy_date_rejects_cross_combined_date(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청기한": "2026-09-01 ~ 2026-09-30",
                "시행일": "2026-10-01",
            },
        }
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "신청기한",
                    "complaint_text": "신청은 2026년 10월 30일까지라고 들었습니다.",
                    "dialogue": "10월 30일까지 기다려야 하는지 궁금합니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, self.persona, policy)
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_DATE_FACT") for error in errors)
        )

    def test_unknown_personal_housing_status_cannot_be_invented(self) -> None:
        persona = {
            **self.persona,
            "family_persona": "가족과 함께 서울에서 생활합니다.",
        }
        result = {
            **self.valid_result,
            "complaints": [
                {
                    "basis": "제외조건",
                    "complaint_text": "주택 소유자 제외 조건의 세부 기준이 궁금합니다.",
                    "dialogue": "저는 주택을 소유하지 않아서 대상이 되어야 합니다.",
                }
            ],
        }

        errors = validate_semantic_quality(result, persona, self.policy)
        self.assertTrue(
            any(error.startswith("UNSUPPORTED_PERSONA_FACT:주거") for error in errors)
        )

    def test_documented_excel_skill_rejects_inability_claim(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "엑셀을 전혀 다룰 줄 모릅니다.",
                    "dialogue": "저는 엑셀도 못 다뤄서 신청이 힘들어요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("PERSONA_FACT_CONTRADICTION:역량") for error in errors)
        )

    def test_self_owned_home_rejects_personal_rent_claim(self) -> None:
        errors = self.errors_for(
            complaints=[
                {
                    "basis": "개인상황",
                    "complaint_text": "제가 살고 있는 집은 월세입니다.",
                    "dialogue": "나는 매달 월세를 내고 있어서 부담이 커요.",
                }
            ]
        )
        self.assertTrue(
            any(error.startswith("PERSONA_FACT_CONTRADICTION:주거") for error in errors)
        )

    def test_summary_must_match_occupation_sex_and_residence(self) -> None:
        errors = self.errors_for(
            persona_summary={
                "이름": "김민지",
                "직업": "개발자",
                "성별": "남자",
                "나이": "25",
                "거주지": "부산-해운대구",
            }
        )
        self.assertIn("PERSONA_SUMMARY_MISMATCH:직업", errors)
        self.assertIn("PERSONA_SUMMARY_MISMATCH:성별", errors)
        self.assertIn("PERSONA_SUMMARY_MISMATCH:거주지", errors)

    def test_boundary_persona_allows_age_exclusion_reason(self) -> None:
        boundary_persona = {**self.persona, "age": 18}
        result = {
            **self.valid_result,
            "persona_summary": {**self.valid_result["persona_summary"], "나이": "18"},
            "grounding": build_grounding_facts(boundary_persona, self.policy),
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "18세라서 19세 나이 하한을 충족하지 못합니다.",
                    "dialogue": "18세는 나이 조건에서 제외돼서 한 살 차이가 아쉬워요.",
                }
            ],
        }
        self.assertEqual(
            build_grounding_facts(boundary_persona, self.policy)["age_status"],
            NOT_MATCHED,
        )
        self.assertEqual(
            validate_semantic_quality(result, boundary_persona, self.policy),
            [],
        )

    def test_nationwide_policy_rejects_region_exclusion_claim(self) -> None:
        nationwide = {
            **self.policy,
            "region_scope": "nationwide",
            "region_province": "",
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, nationwide),
            "complaints": [
                {
                    "basis": "지원대상",
                    "complaint_text": "거주 지역 조건 때문에 지원 대상에서 제외됩니다.",
                    "dialogue": "서울이라는 지역 때문에 지원을 못 받는 건 부당해요.",
                }
            ],
        }
        self.assertEqual(
            build_grounding_facts(self.persona, nationwide)["region_status"],
            MATCHED,
        )
        errors = validate_semantic_quality(result, self.persona, nationwide)
        self.assertTrue(
            any(error.startswith("REGION_ELIGIBILITY") for error in errors)
        )

    def test_explicit_online_method_allows_online_accessibility_complaint(self) -> None:
        policy = {
            **self.policy,
            "상세정보": {
                **self.policy["상세정보"],
                "신청방법": "복지 누리집에서 온라인 신청",
            },
        }
        result = {
            **self.valid_result,
            "grounding": build_grounding_facts(self.persona, policy),
            "complaints": [
                {
                    "basis": "신청방법",
                    "complaint_text": "온라인 신청 과정의 접근성을 개선해 주세요.",
                    "dialogue": "복지 누리집에서 신청할 때 안내를 더 쉽게 보여 주세요.",
                }
            ],
        }
        self.assertEqual(validate_semantic_quality(result, self.persona, policy), [])

    def test_other_condition_uncertainty_is_not_age_contradiction(self) -> None:
        sentences = (
            "나이 기준은 충족하지만 소득 기준이 있는지 알 수 없습니다.",
            "나이 기준에는 맞지만 소득 조건 때문에 지원 대상에서 제외되는지 궁금합니다.",
            "연령 기준에 해당하지만 추가 선정 기준 때문에 지원 대상에서 제외되는지 궁금합니다.",
            "연령 제한에는 문제가 없지만 다른 제외조건 때문에 지원 대상에서 제외되는지 궁금합니다.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "정보미제공",
                            "complaint_text": sentence,
                            "dialogue": sentence,
                        }
                    ],
                }
                errors = validate_semantic_quality(
                    result,
                    self.persona,
                    self.policy,
                )
                self.assertFalse(
                    any(error.startswith("AGE_ELIGIBILITY") for error in errors)
                )

    def test_other_condition_uncertainty_is_not_region_contradiction(self) -> None:
        sentences = (
            "서울 거주 조건은 충족하지만 다른 제외조건 때문에 지원 대상에서 제외되는지 궁금합니다.",
            "지역 조건을 충족하지만 추가 기준 때문에 지원 대상에서 제외되는지 궁금합니다.",
            "지역 조건에는 맞지만 소득 조건 때문에 지원 대상에서 제외되는지 궁금합니다.",
            "서울 거주자로 지역 기준에는 맞지만 다른 제외조건 때문에 지원 대상에서 제외되는지 궁금합니다.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                result = {
                    **self.valid_result,
                    "complaints": [
                        {
                            "basis": "제외조건",
                            "complaint_text": sentence,
                            "dialogue": sentence,
                        }
                    ],
                }

                errors = validate_semantic_quality(result, self.persona, self.policy)

                self.assertFalse(
                    any(error.startswith("REGION_ELIGIBILITY") for error in errors)
                )

    def test_duplicate_complaints_are_rejected(self) -> None:
        complaint = self.valid_result["complaints"][0]
        errors = self.errors_for(complaints=[complaint, dict(complaint)])
        self.assertTrue(any(error.startswith("DUPLICATE_COMPLAINT") for error in errors))


if __name__ == "__main__":
    unittest.main()
