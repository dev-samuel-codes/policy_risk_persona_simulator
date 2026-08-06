import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import api


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

    def test_direct_policy_is_passed_to_simulation_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            simulation_dir = Path(directory) / "simulations"
            with (
                patch.object(api, "SIMULATION_DIR", simulation_dir),
                patch.object(api, "save_active_policy"),
                patch.object(
                    api,
                    "search_similar_policies",
                    return_value=self.similarity,
                ),
                patch.object(api.simulation_executor, "submit") as submit,
            ):
                response = api.set_direct_policy(self.payload)

            submitted_function, submitted_job_id, submitted_policy = (
                submit.call_args.args
            )

        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["job_id"], submitted_job_id)
        self.assertEqual(response["similar_policies"], self.similarity["results"])
        self.assertIs(submitted_function, api.run_simulation_job)
        self.assertEqual(submitted_policy, response["policy"])
        self.assertEqual(
            submitted_policy["상세정보"]["서비스명"],
            "청년 주거 지원",
        )

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
        expected_result = {"policy": policy, "risk_score": {"score": 42.0}}

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


if __name__ == "__main__":
    unittest.main()
