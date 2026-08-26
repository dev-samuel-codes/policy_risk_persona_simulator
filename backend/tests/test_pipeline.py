import unittest
from unittest.mock import patch

from backend.ai_simulation_core import pipeline


class PipelineResultTest(unittest.TestCase):
    def setUp(self) -> None:
        def no_match_batch(items: list[dict]) -> list[dict]:
            return [
                {
                    "status": "no_reliable_match",
                    "results": [],
                    "source": "test_fixture",
                }
                for _ in items
            ]

        search_patcher = patch.object(
            pipeline,
            "find_similar_complaint_cases_batch",
            side_effect=no_match_batch,
        )
        self.complaint_search = search_patcher.start()
        self.addCleanup(search_patcher.stop)

    def test_citizen_result_keeps_selected_persona(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 주거 지원"}}
        citizen_persona = {
            "uuid": "citizen-1",
            "occupation": "조사 전문가",
            "age": 43,
            "province": "경기",
            "district": "성남시 분당구",
        }
        civil_servant_persona = {
            "uuid": "official-1",
            "occupation": "일반 행정 공무원",
            "age": 38,
            "province": "전북",
            "district": "전주시",
        }
        citizen_result = {
            "persona_id": "citizen-1",
            "persona_summary": {"이름": "김진훈", "나이": "43"},
            "personality": "현실적인 주거비 부담을 걱정한다.",
            "complaints": [
                {
                    "complaint_text": "지원 연령에서 제외된다.",
                    "dialogue": "저는 지원받을 수 없어서 답답합니다.",
                }
            ],
        }

        with (
            patch.object(
                pipeline,
                "get_citizen_persona",
                return_value=[citizen_persona],
            ),
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[civil_servant_persona],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value=citizen_result,
            ),
        ):
            result = pipeline.run_pipeline(
                policy=policy,
                citizen_personas=[citizen_persona],
            )

        saved_citizen = result["citizen_results"][0]
        self.assertEqual(saved_citizen["persona"], citizen_persona)
        self.assertEqual(
            saved_citizen["complaints"][0]["dialogue"],
            "저는 지원받을 수 없어서 답답합니다.",
        )
        self.assertNotIn("risk_score", result)
        self.assertEqual(
            result["civil_servant_results"][0]["persona"],
            civil_servant_persona,
        )
        self.assertEqual(
            saved_citizen["complaints"][0]["precedent_search"]["status"],
            "no_reliable_match",
        )
        self.assertEqual(saved_citizen["complaints"][0]["reference_cases"], [])
        self.assertNotIn("realism_check", result)

    def test_reference_search_runs_once_in_complaint_order_with_context(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}
        first_persona = {"uuid": "first", "age": 29, "province": "경기"}
        second_persona = {"uuid": "second", "age": 35, "province": "경기"}
        first_complaint = {
            "complaint_text": "  지원 대상에서 제외됩니다.  ",
            "dialogue": "첫 번째 대화",
        }
        fallback_complaint = {
            "complaint_text": " ",
            "dialogue": "대화 내용을 검색어로 사용해 주세요.",
        }
        simulation_results = [
            {
                "persona": first_persona,
                "complaints": [first_complaint],
            },
            {
                "persona": second_persona,
                "complaints": [fallback_complaint],
            },
        ]
        matched_case = {
            "title": "경기도 청년 주거 지원",
            "match_score": 84.0,
            "reference_eligible": True,
        }
        self.complaint_search.side_effect = None
        self.complaint_search.return_value = [
            {
                "status": "matched",
                "results": [matched_case],
                "source": "public_faq_snapshot",
                "index_version": "test-v1",
            },
            {
                "status": "no_reliable_match",
                "results": [],
                "source": "public_faq_snapshot",
                "index_version": "test-v1",
            },
        ]

        pipeline.attach_complaint_reference_cases(
            simulation_results,
            policy=policy,
        )

        self.complaint_search.assert_called_once_with(
            [
                {
                    "complaint_text": "지원 대상에서 제외됩니다.",
                    "policy": policy,
                    "persona": first_persona,
                },
                {
                    "complaint_text": "대화 내용을 검색어로 사용해 주세요.",
                    "policy": policy,
                    "persona": second_persona,
                },
            ]
        )
        self.assertEqual(first_complaint["reference_cases"], [matched_case])
        self.assertNotIn("results", first_complaint["precedent_search"])
        self.assertEqual(
            first_complaint["precedent_search"]["source"],
            "public_faq_snapshot",
        )
        self.assertEqual(fallback_complaint["reference_cases"], [])
        self.assertEqual(
            fallback_complaint["precedent_search"]["status"],
            "no_reliable_match",
        )

    def test_reference_summary_separates_all_search_outcomes(self) -> None:
        results = [
            {
                "complaints": [
                    {"precedent_search": {"status": "matched"}},
                    {"precedent_search": {"status": "no_reliable_match"}},
                    {"precedent_search": {"status": "unavailable"}},
                    {"precedent_search": {"status": "invalid_query"}},
                ]
            }
        ]

        summary = pipeline.compute_complaint_reference_summary(results)

        self.assertEqual(
            summary,
            {
                "total": 4,
                "evaluated": 2,
                "matched": 1,
                "unavailable": 1,
                "invalid": 1,
                "reference_rate": 50.0,
                "search_coverage": 50.0,
                "status": "partial",
            },
        )

    def test_reference_summary_does_not_report_outage_as_zero_match_rate(
        self,
    ) -> None:
        unavailable_results = [
            {
                "complaints": [
                    {"precedent_search": {"status": "unavailable"}},
                    {"precedent_search": {"status": "invalid_query"}},
                ]
            }
        ]

        summary = pipeline.compute_complaint_reference_summary(unavailable_results)

        self.assertEqual(summary["evaluated"], 0)
        self.assertIsNone(summary["reference_rate"])
        self.assertEqual(summary["search_coverage"], 0.0)
        self.assertEqual(summary["status"], "unavailable")

    def test_empty_query_is_invalid_and_not_sent_to_search(self) -> None:
        complaint = {"complaint_text": " ", "dialogue": ""}

        pipeline.attach_complaint_reference_cases(
            [{"persona": {"uuid": "empty"}, "complaints": [complaint]}],
            policy={"상세정보": {"서비스명": "테스트 정책"}},
        )

        self.complaint_search.assert_called_once_with([])
        self.assertEqual(complaint["reference_cases"], [])
        self.assertEqual(
            complaint["precedent_search"],
            {
                "status": "invalid_query",
                "reason_code": "empty_complaint_text",
            },
        )

    def test_reference_search_exceptions_degrade_without_losing_invalid_status(
        self,
    ) -> None:
        exceptions = [
            (
                pipeline.CivilComplaintIndexUnavailableError("index missing"),
                "index_unavailable",
            ),
            (RuntimeError("search crashed"), "search_failed"),
        ]

        for error, reason_code in exceptions:
            with self.subTest(error=type(error).__name__):
                valid_complaint = {"complaint_text": "지원 절차가 어렵습니다."}
                invalid_complaint = {"complaint_text": "", "dialogue": " "}
                self.complaint_search.reset_mock()
                self.complaint_search.side_effect = error

                pipeline.attach_complaint_reference_cases(
                    [
                        {
                            "persona": {"uuid": "citizen-1"},
                            "complaints": [valid_complaint, invalid_complaint],
                        }
                    ],
                    policy={"상세정보": {"서비스명": "테스트 정책"}},
                )

                self.complaint_search.assert_called_once()
                self.assertEqual(valid_complaint["reference_cases"], [])
                self.assertEqual(
                    valid_complaint["precedent_search"],
                    {"status": "unavailable", "reason_code": reason_code},
                )
                self.assertEqual(
                    invalid_complaint["precedent_search"]["status"],
                    "invalid_query",
                )

    def test_matched_response_requires_an_eligible_reference_case(self) -> None:
        complaint = {"complaint_text": "월세 지원 대상에서 제외됩니다."}
        self.complaint_search.side_effect = None
        self.complaint_search.return_value = [
            {
                "status": "matched",
                "results": [{"case_id": "hidden", "reference_eligible": False}],
            }
        ]

        pipeline.attach_complaint_reference_cases(
            [{"persona": {"uuid": "citizen-1"}, "complaints": [complaint]}],
            policy={"상세정보": {"서비스명": "청년 월세 지원"}},
        )

        self.assertEqual(complaint["reference_cases"], [])
        self.assertEqual(
            complaint["precedent_search"],
            {
                "status": "unavailable",
                "reason_code": "invalid_search_response",
            },
        )

    def test_explicit_citizen_personas_run_in_exact_order(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}
        selected = [
            {"uuid": "eligible", "age": 19},
            {"uuid": "lower-boundary", "age": 18},
            {"uuid": "upper-boundary", "age": 35},
        ]

        def citizen_result(persona: dict, policy: dict) -> dict:
            return {
                "persona_id": persona["uuid"],
                "complaints": [{"dialogue": persona["uuid"]}],
            }

        with (
            patch.object(pipeline, "get_citizen_persona") as random_citizens,
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[
                    {"uuid": "official-1", "occupation": "일반 행정 공무원"},
                    {"uuid": "official-2", "occupation": "일반 행정 공무원"},
                    {"uuid": "official-3", "occupation": "일반 행정 공무원"},
                ],
            ) as officials,
            patch.object(
                pipeline,
                "run_citizen_simulation",
                side_effect=citizen_result,
            ),
            patch.object(pipeline, "unload_llm"),
        ):
            result = pipeline.run_pipeline(
                policy=policy,
                citizen_personas=selected,
            )

        random_citizens.assert_not_called()
        officials.assert_called_once_with(
            limit=3,
            keyword="공무원",
            min_age=20,
            max_age=60,
        )
        self.assertEqual(
            [item["persona_id"] for item in result["citizen_results"]],
            ["eligible", "lower-boundary", "upper-boundary"],
        )
        self.assertEqual(len(result["civil_servant_results"]), 3)
        self.complaint_search.assert_called_once()
        search_items = self.complaint_search.call_args.args[0]
        self.assertEqual(
            [item["complaint_text"] for item in search_items],
            ["eligible", "lower-boundary", "upper-boundary"],
        )
        self.assertEqual(
            [item["persona"]["uuid"] for item in search_items],
            ["eligible", "lower-boundary", "upper-boundary"],
        )
        self.assertTrue(all(item["policy"] is policy for item in search_items))

    def test_explicit_persona_failure_fails_the_whole_pipeline(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}
        selected = [{"uuid": "selected-1", "age": 19}]

        with (
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[
                    {"uuid": "official-1", "occupation": "일반 행정 공무원"}
                ],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value=None,
            ),
            patch.object(pipeline, "unload_llm") as unload,
        ):
            with self.assertRaisesRegex(RuntimeError, "selected-1"):
                pipeline.run_pipeline(
                    policy=policy,
                    citizen_personas=selected,
                )

        unload.assert_called_once_with()
        self.complaint_search.assert_not_called()

    def test_search_failure_does_not_fail_pipeline(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}
        selected = [{"uuid": "selected-1", "age": 29}]
        self.complaint_search.side_effect = RuntimeError("temporary outage")

        with (
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[
                    {"uuid": "official-1", "occupation": "일반 행정 공무원"}
                ],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value={
                    "persona_id": "selected-1",
                    "complaints": [
                        {
                            "complaint_text": "월세 지원을 받을 수 없습니다.",
                            "dialogue": "왜 저는 대상이 아닌가요?",
                        }
                    ],
                },
            ),
            patch.object(pipeline, "unload_llm"),
        ):
            result = pipeline.run_pipeline(
                policy=policy,
                citizen_personas=selected,
            )

        complaint = result["citizen_results"][0]["complaints"][0]
        self.assertEqual(complaint["precedent_search"]["status"], "unavailable")
        self.assertNotIn("risk_score", result)
        self.assertNotIn("risk_category", complaint)
        self.assertEqual(
            result["complaint_reference_summary"],
            {
                "total": 1,
                "evaluated": 0,
                "matched": 0,
                "unavailable": 1,
                "invalid": 0,
                "reference_rate": None,
                "search_coverage": 0.0,
                "status": "unavailable",
            },
        )

    def test_implicit_persona_failure_fails_closed_before_downstream_steps(
        self,
    ) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}

        with (
            patch.object(
                pipeline,
                "get_citizen_persona",
                return_value=[
                    {"uuid": "random-1"},
                    {"uuid": "random-2"},
                    {"uuid": "random-3"},
                ],
            ),
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[
                    {"uuid": "official-1", "occupation": "일반 행정 공무원"},
                    {"uuid": "official-2", "occupation": "일반 행정 공무원"},
                    {"uuid": "official-3", "occupation": "일반 행정 공무원"},
                ],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value=None,
            ),
            patch.object(pipeline, "run_civil_servant_simulation") as official_run,
            patch.object(pipeline, "unload_llm"),
        ):
            with self.assertRaisesRegex(RuntimeError, "random-1"):
                pipeline.run_pipeline(policy=policy)

        official_run.assert_not_called()
        self.complaint_search.assert_not_called()

    def test_invalid_official_response_fails_closed_before_downstream_steps(
        self,
    ) -> None:
        policy = {"상세정보": {"서비스명": "청년 월세 지원"}}
        selected = [{"uuid": "selected-1", "age": 29}]

        with (
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[
                    {"uuid": "official-1", "occupation": "일반 행정 공무원"}
                ],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value={
                    "persona_id": "selected-1",
                    "complaints": [
                        {
                            "basis": "개인상황",
                            "complaint_text": "제 상황에서 신청 가능한지 궁금합니다.",
                        }
                    ],
                },
            ),
            patch.object(
                pipeline,
                "run_civil_servant_simulation",
                return_value={
                    "official_persona_id": "official-1",
                    "citizen_persona_id": "selected-1",
                    "basis": "개인상황",
                    "response": "이미 승인되었습니다.",
                    "_validation_errors": [],
                    "_quality_gate": {
                        "status": "passed",
                        "mode": "deterministic_policy_grounded_v1",
                        "removed_statements": 0,
                    },
                },
            ),
            patch.object(pipeline, "unload_llm"),
        ):
            with self.assertRaisesRegex(RuntimeError, "공무원 페르소나 응답"):
                pipeline.run_pipeline(
                    policy=policy,
                    citizen_personas=selected,
                )

        self.complaint_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
