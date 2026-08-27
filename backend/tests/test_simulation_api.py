import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import api
from backend.ai_simulation_core.simulations.civil_servant_simulation import (
    QUALITY_MODE,
    build_grounded_response,
)


class SimulationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = api.DirectPolicyInput(
            policy_name="청년 주거 지원",
            target_audience="만 19세 이상 34세 이하 청년",
            benefits="월 20만 원 지원",
        )
        self.similarity = {
            "as_of_date": "2026-08-06",
            "index_version": "2026-08-06T00:00:00+00:00",
            "source_count": 10966,
            "query_time_ms": 25.0,
            "results": [
                {
                    "service_id": "past-policy-1",
                    "policy_name": "기존 청년 주거 지원",
                    "similarity_score": 88.0,
                }
            ],
        }

    def _completed_result(self, persona_ids: list[str]) -> dict:
        policy = {
            "상세정보": {
                "서비스명": "청년 주거 지원",
                "지원내용": "월 20만 원 지원",
            }
        }
        citizen_results = []
        official_results = []
        for index, persona_id in enumerate(persona_ids, start=1):
            citizen_persona = {
                "uuid": persona_id,
                "occupation": "사무원",
                "age": 25,
                "province": "서울",
                "district": "서울-서초구",
            }
            citizen_result = {
                "persona_id": persona_id,
                "persona": citizen_persona,
                "complaints": [
                    {
                        "basis": "개인상황",
                        "complaint_text": "제 상황에서 신청 가능한지 궁금합니다.",
                        "dialogue": "전체 자격과 승인 여부를 확인하고 싶습니다.",
                    }
                ],
                "_validation_errors": [],
                "_quality_gate": {"status": "passed"},
            }
            official_persona = {
                "uuid": f"official-{index}",
                "occupation": "일반 행정 공무원",
                "age": 40,
                "province": "서울",
                "district": "서울-중구",
            }
            basis, response = build_grounded_response(policy, citizen_result)
            official_result = {
                "official_persona_id": official_persona["uuid"],
                "citizen_persona_id": persona_id,
                "basis": basis,
                "response": response,
                "_validation_errors": [],
                "_quality_gate": {
                    "status": "passed",
                    "mode": QUALITY_MODE,
                    "removed_statements": 0,
                    "generation_attempts": 1,
                },
            }
            citizen_results.append(citizen_result)
            official_results.append(
                {
                    "persona_index": index,
                    "persona": official_persona,
                    **official_result,
                }
            )
        return {
            "policy": policy,
            "citizen_results": citizen_results,
            "civil_servant_results": official_results,
        }

    def test_direct_policy_uses_filtered_random_simulation_path(self) -> None:
        payload = api.DirectPolicyInput(
            policy_name="서울 청년 월세 지원",
            benefits="월 20만 원 지원",
            region_scope="specific",
            region_province="서울",
            region_district="서울-서초구",
            age_min=19,
            age_max=34,
        )
        candidates = [
            {
                "uuid": f"direct-{index}",
                "occupation": "사무원",
                "province": "서울",
                "district": "서울-서초구",
                "age": 20 + index,
                "match": {
                    "region_match": True,
                    "age_cohort": "eligible",
                    "age_match_reason": "within_range",
                },
            }
            for index in range(3)
        ]
        personas = [
            {key: value for key, value in candidate.items() if key != "match"}
            for candidate in candidates
        ]
        matches = [
            {"persona_id": persona["uuid"], **candidate["match"]}
            for persona, candidate in zip(personas, candidates, strict=True)
        ]

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            with (
                patch.object(api, "SIMULATION_DIR", simulation_dir),
                patch.object(api.secrets, "randbits", return_value=4242),
                patch.object(
                    api,
                    "get_persona_candidates",
                    return_value=candidates,
                ) as get_candidates,
                patch.object(
                    api,
                    "validate_persona_selection",
                    return_value=matches,
                ),
                patch.object(api, "save_active_policy"),
                patch.object(
                    api,
                    "search_similar_policies",
                    return_value=self.similarity,
                ),
                patch.object(api.simulation_executor, "submit") as submit,
            ):
                response = api.set_direct_policy(payload)

        get_candidates.assert_called_once_with(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="eligible",
            limit=3,
            seed=4242,
        )
        submitted_function, _, submitted_policy, submitted_personas = (
            submit.call_args.args
        )
        self.assertIs(submitted_function, api.run_simulation_job)
        self.assertEqual(submitted_policy, response["policy"])
        self.assertEqual(submitted_personas, personas)
        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["source"], "direct_input")
        self.assertEqual(response["selection_mode"], "random")
        self.assertEqual(response["selection_seed"], 4242)
        self.assertEqual(response["similar_policies"], self.similarity["results"])
        self.assertIn("similarity", response)

    def test_direct_policy_does_not_queue_with_insufficient_filtered_candidates(
        self,
    ) -> None:
        payload = api.DirectPolicyInput(
            policy_name="서울 청년 월세 지원",
            benefits="월 20만 원 지원",
            region_scope="specific",
            region_province="서울",
            region_district="서울-서초구",
            age_min=19,
            age_max=34,
        )
        with (
            patch.object(api.secrets, "randbits", return_value=7),
            patch.object(
                api,
                "get_persona_candidates",
                return_value=[{"uuid": "only-one", "match": {}}],
            ) as get_candidates,
            patch.object(api, "save_active_policy") as save_active,
            patch.object(api.simulation_executor, "submit") as submit,
        ):
            with self.assertRaisesRegex(api.HTTPException, "3명을 찾지 못했습니다"):
                api.set_direct_policy(payload)

        get_candidates.assert_called_once_with(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="eligible",
            limit=3,
            seed=7,
        )
        save_active.assert_not_called()
        submit.assert_not_called()

    def test_similar_policy_endpoint_returns_ranked_results(self) -> None:
        payload = api.SimilarPolicyInput(
            policy_name="청년 주거 지원",
            target_audience="만 19세 이상 34세 이하 청년",
            benefits="월 20만 원 지원",
            top_k=3,
        )
        with patch.object(
            api,
            "search_similar_policies",
            return_value=self.similarity,
        ) as search:
            response = api.get_similar_policies(payload)

        search.assert_called_once()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["results"], self.similarity["results"])

    def test_simulation_job_saves_completed_result(self) -> None:
        policy = api.build_direct_policy(self.payload.model_dump())
        expected_result = {"policy": policy}

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            job_id = "a426d20f-f196-4f0d-9aee-cc5b29ebbdca"
            with patch.object(api, "SIMULATION_DIR", simulation_dir):
                api.write_simulation_job(
                    {
                        "job_id": job_id,
                        "status": "queued",
                        "created_at": api.utc_now(),
                        "similar_policies": self.similarity["results"],
                        "similarity": api.policy_similarity_metadata(self.similarity),
                    }
                )
                with patch.object(
                    api,
                    "run_pipeline",
                    return_value=expected_result,
                ) as run_pipeline:
                    api.run_simulation_job(job_id, policy)
                saved_job = api.load_simulation_job(job_id)

        run_pipeline.assert_called_once_with(policy=policy)
        self.assertEqual(saved_job["status"], "completed")
        self.assertEqual(saved_job["policy"], policy)
        self.assertEqual(saved_job["similar_policies"], self.similarity["results"])
        self.assertEqual(saved_job["result"], expected_result)
        self.assertIsNone(saved_job["error"])

    def test_load_simulation_job_hides_deprecated_scoring_fields(self) -> None:
        legacy_job = {
            "job_id": "8c1bd23f-240c-4273-b124-ff3450744746",
            "status": "completed",
            "result": {
                "risk_score": {"score": 42.0},
                "citizen_results": [
                    {
                        "complaints": [
                            {
                                "complaint_text": "지원 대상에서 제외됩니다.",
                                "risk_category": "target_ambiguous",
                            }
                        ]
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            with patch.object(api, "SIMULATION_DIR", simulation_dir):
                api.write_simulation_job(legacy_job)
                loaded_job = api.load_simulation_job(legacy_job["job_id"])

        self.assertNotIn("risk_score", loaded_job["result"])
        complaint = loaded_job["result"]["citizen_results"][0]["complaints"][0]
        self.assertNotIn("risk_category", complaint)
        self.assertEqual(
            complaint["complaint_text"],
            "지원 대상에서 제외됩니다.",
        )

    def test_simulation_job_saves_failure(self) -> None:
        policy = api.build_direct_policy(self.payload.model_dump())

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            job_id = "85033683-745f-4b90-bc07-9fe544f30c95"
            with patch.object(api, "SIMULATION_DIR", simulation_dir):
                api.write_simulation_job(
                    {
                        "job_id": job_id,
                        "status": "queued",
                        "created_at": api.utc_now(),
                    }
                )
                with patch.object(
                    api,
                    "run_pipeline",
                    side_effect=RuntimeError("모델 실행 오류"),
                ):
                    api.run_simulation_job(job_id, policy)
                saved_job = api.load_simulation_job(job_id)

        self.assertEqual(saved_job["status"], "failed")
        self.assertEqual(saved_job["error"], "모델 실행 오류")
        self.assertIsNone(saved_job["result"])

    def test_persona_simulation_submits_exact_selected_personas(self) -> None:
        payload = api.PersonaSimulationInput(
            policy=api.DirectPolicyInput(
                policy_name="서울 청년 월세 지원",
                target_audience="서울 거주 만 19세 이상 34세 이하 청년",
                benefits="월 20만 원 지원",
                region_scope="specific",
                region_province="서울",
                region_district="서울-서초구",
                age_min=19,
                age_max=34,
            ),
            persona_ids=["eligible", "lower-boundary", "upper-boundary"],
        )
        personas = [
            {
                "uuid": "eligible",
                "occupation": "사무원",
                "province": "서울",
                "district": "서울-서초구",
                "age": 19,
            },
            {
                "uuid": "lower-boundary",
                "occupation": "학생",
                "province": "서울",
                "district": "서울-서초구",
                "age": 18,
            },
            {
                "uuid": "upper-boundary",
                "occupation": "연구원",
                "province": "서울",
                "district": "서울-서초구",
                "age": 35,
            },
        ]
        matches = [
            {
                "persona_id": "eligible",
                "region_match": True,
                "age_cohort": "eligible",
                "age_match_reason": "within_range",
            },
            {
                "persona_id": "lower-boundary",
                "region_match": True,
                "age_cohort": "boundary",
                "age_match_reason": "below_minimum",
            },
            {
                "persona_id": "upper-boundary",
                "region_match": True,
                "age_cohort": "boundary",
                "age_match_reason": "above_maximum",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            with (
                patch.object(api, "SIMULATION_DIR", simulation_dir),
                patch.object(api, "resolve_personas", return_value=personas) as resolve,
                patch.object(
                    api,
                    "validate_persona_selection",
                    return_value=matches,
                ) as validate,
                patch.object(api, "save_active_policy"),
                patch.object(
                    api,
                    "search_similar_policies",
                    return_value=self.similarity,
                ),
                patch.object(api.simulation_executor, "submit") as submit,
            ):
                response = api.create_persona_simulation(payload)
                saved_job = api.load_simulation_job(response["job_id"])

        resolve.assert_called_once_with(payload.persona_ids)
        validate.assert_called_once_with(
            personas,
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
        )
        submitted_function, submitted_job_id, submitted_policy, submitted_personas = (
            submit.call_args.args
        )
        self.assertIs(submitted_function, api.run_simulation_job)
        self.assertEqual(submitted_job_id, response["job_id"])
        self.assertEqual(submitted_policy, response["policy"])
        self.assertEqual(submitted_personas, personas)
        self.assertEqual(saved_job["selection_mode"], "manual")
        self.assertIsNone(saved_job["selection_seed"])
        self.assertEqual(saved_job["persona_ids"], payload.persona_ids)
        self.assertEqual(
            [item["selection_cohort"] for item in saved_job["selected_personas"]],
            ["eligible", "boundary", "boundary"],
        )

    def test_persona_simulation_requires_three_distinct_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "서로 다른"):
            api.PersonaSimulationInput(
                policy=self.payload,
                persona_ids=["same", "same", "third"],
            )

    def test_manual_persona_simulation_requires_exactly_three_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "3명이 필요"):
            api.PersonaSimulationInput(policy=self.payload)

        with self.assertRaisesRegex(ValueError, "3명이 필요"):
            api.PersonaSimulationInput(
                policy=self.payload,
                persona_ids=["one", "two"],
            )

    def test_random_persona_simulation_rejects_explicit_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "함께 지정할 수 없습니다"):
            api.PersonaSimulationInput(
                policy=self.payload,
                selection_mode="random",
                persona_ids=["one", "two", "three"],
            )

    def test_random_persona_simulation_samples_three_eligible_personas(self) -> None:
        policy_input = api.DirectPolicyInput(
            policy_name="서울 청년 월세 지원",
            target_audience="서울 거주 만 19세 이상 34세 이하 청년",
            benefits="월 20만 원 지원",
            region_scope="specific",
            region_province="서울",
            region_district="서울-서초구",
            age_min=19,
            age_max=34,
        )
        payload = api.PersonaSimulationInput(
            policy=policy_input,
            selection_mode="random",
        )
        candidates = [
            {
                "uuid": f"random-{index}",
                "occupation": "사무원",
                "province": "서울",
                "district": "서울-서초구",
                "age": 20 + index,
                "match": {
                    "region_match": True,
                    "age_cohort": "eligible",
                    "age_match_reason": "within_range",
                },
            }
            for index in range(3)
        ]
        personas = [
            {key: value for key, value in candidate.items() if key != "match"}
            for candidate in candidates
        ]
        matches = [
            {"persona_id": persona["uuid"], **candidate["match"]}
            for persona, candidate in zip(personas, candidates, strict=True)
        ]

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            with (
                patch.object(api, "SIMULATION_DIR", simulation_dir),
                patch.object(api.secrets, "randbits", return_value=4242) as randbits,
                patch.object(
                    api,
                    "get_persona_candidates",
                    return_value=candidates,
                ) as get_candidates,
                patch.object(api, "resolve_personas") as resolve,
                patch.object(
                    api,
                    "validate_persona_selection",
                    return_value=matches,
                ) as validate,
                patch.object(api, "save_active_policy"),
                patch.object(
                    api,
                    "search_similar_policies",
                    return_value=self.similarity,
                ),
                patch.object(api.simulation_executor, "submit") as submit,
            ):
                response = api.create_persona_simulation(payload)
                saved_job = api.load_simulation_job(response["job_id"])

        randbits.assert_called_once_with(63)
        get_candidates.assert_called_once_with(
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
            cohort="eligible",
            limit=3,
            seed=4242,
        )
        resolve.assert_not_called()
        validate.assert_called_once_with(
            personas,
            region_scope="specific",
            province="서울",
            district="서울-서초구",
            age_min=19,
            age_max=34,
        )
        submitted_personas = submit.call_args.args[3]
        self.assertEqual(submitted_personas, personas)
        self.assertEqual(response["selection_mode"], "random")
        self.assertEqual(response["selection_seed"], 4242)
        self.assertEqual(response["persona_ids"], [item["uuid"] for item in personas])
        self.assertEqual(saved_job["selection_mode"], "random")
        self.assertEqual(saved_job["selection_seed"], 4242)
        self.assertTrue(
            all(
                item["selection_cohort"] == "eligible"
                for item in saved_job["selected_personas"]
            )
        )

    def test_random_persona_simulation_requires_three_matching_candidates(self) -> None:
        payload = api.PersonaSimulationInput(
            policy=self.payload,
            selection_mode="random",
        )

        with (
            patch.object(api.secrets, "randbits", return_value=7),
            patch.object(
                api,
                "get_persona_candidates",
                return_value=[{"uuid": "only-one", "match": {}}],
            ),
            patch.object(api.simulation_executor, "submit") as submit,
        ):
            with self.assertRaisesRegex(api.HTTPException, "3명을 찾지 못했습니다"):
                api.create_persona_simulation(payload)

        submit.assert_not_called()

    def test_nationwide_policy_rejects_specific_region_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "전국 정책"):
            api.DirectPolicyInput(
                policy_name="전국 청년 월세 지원",
                benefits="월 20만 원 지원",
                region_scope="nationwide",
                region_province="서울",
            )

        with self.assertRaisesRegex(ValueError, "전국 정책"):
            api.DirectPolicyInput(
                policy_name="전국 청년 월세 지원",
                benefits="월 20만 원 지원",
                region_scope="nationwide",
                region_district="서울-서초구",
            )

    def test_selected_personas_and_matching_evidence_survive_job_updates(self) -> None:
        policy = api.build_direct_policy(self.payload.model_dump())
        personas = [{"uuid": "one"}, {"uuid": "two"}, {"uuid": "three"}]
        selected_snapshots = [
            {**persona, "selection_cohort": "eligible"} for persona in personas
        ]
        matches = [
            {
                "persona_id": persona["uuid"],
                "region_match": True,
                "age_cohort": "eligible",
                "age_match_reason": "no_age_restriction",
            }
            for persona in personas
        ]
        expected_result = {
            "citizen_results": [{"persona_id": persona["uuid"]} for persona in personas]
        }

        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            job_id = "3ba5011e-7490-42c2-aa8a-e06e24118773"
            with patch.object(api, "SIMULATION_DIR", simulation_dir):
                api.write_simulation_job(
                    {
                        "job_id": job_id,
                        "status": "queued",
                        "created_at": api.utc_now(),
                        "selected_personas": selected_snapshots,
                        "selection_match": matches,
                    }
                )
                with patch.object(
                    api,
                    "run_pipeline",
                    return_value=expected_result,
                ) as run_pipeline:
                    api.run_simulation_job(job_id, policy, personas)
                saved_job = api.load_simulation_job(job_id)

        run_pipeline.assert_called_once_with(
            policy=policy,
            citizen_personas=personas,
        )
        self.assertEqual(saved_job["status"], "completed")
        self.assertEqual(saved_job["selected_personas"], selected_snapshots)
        self.assertEqual(saved_job["selection_match"], matches)

    def test_completed_result_integrity_accepts_exact_three_personas(self) -> None:
        persona_ids = ["one", "two", "three"]
        result = self._completed_result(persona_ids)

        api.validate_completed_simulation_result(result, persona_ids)

    def test_completed_result_accepts_nonempty_official_response_without_content_gate(
        self,
    ) -> None:
        persona_ids = ["one", "two", "three"]
        result = self._completed_result(persona_ids)
        result["civil_servant_results"][0]["response"] = "이미 승인되었습니다."

        api.validate_completed_simulation_result(result, persona_ids)

    def test_completed_result_integrity_rejects_inactive_official(self) -> None:
        persona_ids = ["one", "two", "three"]
        result = self._completed_result(persona_ids)
        result["civil_servant_results"][0]["persona"]["occupation"] = (
            "전직 중앙정부 고위 공무원, 현재 구직중"
        )

        with self.assertRaisesRegex(RuntimeError, "공무원 응답 1"):
            api.validate_completed_simulation_result(result, persona_ids)

    def test_completed_result_integrity_rejects_invalid_official_quality_gate(
        self,
    ) -> None:
        persona_ids = ["one", "two", "three"]
        cases = [
            {
                "name": "wrong-mode",
                "field": "mode",
                "value": "deterministic_policy_grounded_v1",
                "delete": False,
            },
            {
                "name": "missing-generation-attempts",
                "field": "generation_attempts",
                "value": None,
                "delete": True,
            },
            {
                "name": "zero-generation-attempts",
                "field": "generation_attempts",
                "value": 0,
                "delete": False,
            },
            {
                "name": "too-many-generation-attempts",
                "field": "generation_attempts",
                "value": 4,
                "delete": False,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                result = self._completed_result(persona_ids)
                gate = result["civil_servant_results"][0]["_quality_gate"]
                if case["delete"]:
                    gate.pop(case["field"])
                else:
                    gate[case["field"]] = case["value"]

                with self.assertRaisesRegex(
                    RuntimeError,
                    "OFFICIAL_QUALITY_GATE_INVALID",
                ):
                    api.validate_completed_simulation_result(result, persona_ids)

    def test_completed_result_integrity_rejects_official_link_mutations(
        self,
    ) -> None:
        persona_ids = ["one", "two", "three"]

        def swap_citizen_links(result: dict) -> None:
            officials = result["civil_servant_results"]
            officials[0]["citizen_persona_id"], officials[1][
                "citizen_persona_id"
            ] = (
                officials[1]["citizen_persona_id"],
                officials[0]["citizen_persona_id"],
            )

        def mismatch_official_persona_id(result: dict) -> None:
            result["civil_servant_results"][0][
                "official_persona_id"
            ] = "official-other"

        cases = [
            {
                "name": "official-citizen-order-swapped",
                "mutate": swap_citizen_links,
                "error": "OFFICIAL_CITIZEN_LINK_MISMATCH",
            },
            {
                "name": "official-id-does-not-match-nested-persona",
                "mutate": mismatch_official_persona_id,
                "error": "OFFICIAL_PERSONA_ID_MISMATCH",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                result = self._completed_result(persona_ids)
                case["mutate"](result)

                with self.assertRaisesRegex(RuntimeError, case["error"]):
                    api.validate_completed_simulation_result(result, persona_ids)

    def test_completed_result_integrity_rejects_partial_result(self) -> None:
        persona_ids = ["one", "two", "three"]
        partial_result = {
            "citizen_results": [
                {
                    "persona_id": "one",
                    "persona": {"uuid": "one"},
                    "_validation_errors": [],
                    "_quality_gate": {"status": "passed"},
                }
            ],
            "civil_servant_results": [{}],
        }

        with self.assertRaisesRegex(RuntimeError, "시민 응답 수"):
            api.validate_completed_simulation_result(partial_result, persona_ids)


if __name__ == "__main__":
    unittest.main()
