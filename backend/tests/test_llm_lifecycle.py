import unittest
from unittest.mock import Mock, patch

from backend.ai_simulation_core import pipeline
from backend.ai_simulation_core.llm import llm_gateway
from backend.ai_simulation_core.llm.qwen_model import LLM


def _fake_llm_worker(connection) -> None:
    connection.send(("ready", None))
    try:
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                return
            connection.send(("response", f"응답:{payload}"))
    finally:
        connection.close()


class LLMGatewayLifecycleTests(unittest.TestCase):
    def tearDown(self) -> None:
        llm_gateway._llm = None

    def test_unload_llm_closes_and_clears_shared_instance(self) -> None:
        fake_llm = Mock()
        llm_gateway._llm = fake_llm

        llm_gateway.unload_llm()

        fake_llm.close.assert_called_once_with()
        self.assertIsNone(llm_gateway._llm)

    def test_process_client_exits_after_close(self) -> None:
        client = llm_gateway.LLMProcessClient(worker_target=_fake_llm_worker)
        process = client._process

        self.assertTrue(process.is_alive())
        self.assertEqual(client.generate("테스트"), "응답:테스트")

        client.close()

        self.assertFalse(process.is_alive())

    def test_unload_llm_is_safe_before_model_load(self) -> None:
        llm_gateway._llm = None
        llm_gateway.unload_llm()
        self.assertIsNone(llm_gateway._llm)


class LLMCleanupTests(unittest.TestCase):
    @patch("backend.ai_simulation_core.llm.qwen_model.gc.collect")
    @patch("backend.ai_simulation_core.llm.qwen_model.torch")
    def test_close_releases_cuda_model_and_is_idempotent(
        self, torch_mock: Mock, collect_mock: Mock
    ) -> None:
        torch_mock.cuda.is_available.return_value = True
        torch_mock.backends.mps.is_available.return_value = False
        llm = LLM.__new__(LLM)
        llm.model = Mock()
        llm.tokenizer = Mock()
        llm._closed = False

        llm.close()
        llm.close()

        self.assertIsNone(llm.model)
        self.assertIsNone(llm.tokenizer)
        collect_mock.assert_called_once_with()
        torch_mock.cuda.synchronize.assert_called_once_with()
        torch_mock.cuda.empty_cache.assert_called_once_with()


class PipelineLifecycleTests(unittest.TestCase):
    @patch.object(pipeline, "unload_llm")
    @patch.object(pipeline, "_run_pipeline")
    def test_pipeline_unloads_after_success(
        self, run_mock: Mock, unload_mock: Mock
    ) -> None:
        expected = {"risk_score": {"score": 1}}
        run_mock.return_value = expected

        self.assertEqual(pipeline.run_pipeline(), expected)
        unload_mock.assert_called_once_with()

    @patch.object(pipeline, "unload_llm")
    @patch.object(pipeline, "_run_pipeline", side_effect=RuntimeError("실패"))
    def test_pipeline_unloads_after_failure(
        self, run_mock: Mock, unload_mock: Mock
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "실패"):
            pipeline.run_pipeline()

        unload_mock.assert_called_once_with()

    @patch.object(pipeline, "unload_llm")
    @patch.object(pipeline, "_run_pipeline", side_effect=KeyboardInterrupt)
    def test_pipeline_unloads_after_keyboard_interrupt(
        self, run_mock: Mock, unload_mock: Mock
    ) -> None:
        with self.assertRaises(KeyboardInterrupt):
            pipeline.run_pipeline()

        unload_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
