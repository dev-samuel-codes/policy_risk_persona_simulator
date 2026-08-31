import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from backend.ai_simulation_core.llm.qwen_model import (
    CPU_MAX_MEMORY,
    CUDA_MAX_MEMORY,
    GIB,
    LLM,
    QWEN_4B_MODEL_NAME,
    QWEN_8B_MODEL_NAME,
    QWEN_MODEL_NAME_ENV,
    SystemResources,
    get_cuda_max_memory,
    get_system_resources,
    select_qwen_model,
)


class _ModelInputs(dict):
    def __init__(self) -> None:
        input_ids = torch.tensor([[10, 11]])
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "templated"

    def __call__(self, *_args, **_kwargs) -> _ModelInputs:
        return _ModelInputs()

    def batch_decode(self, generated_ids, **_kwargs) -> list[str]:
        self.generated_ids = generated_ids
        return ["응답"]


class _Model:
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 20, 21]])


class QwenModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda_resources = SystemResources(
            device_type="cuda",
            cpu_count=16,
            system_memory_total=32 * GIB,
            system_memory_available=28 * GIB,
            cuda_memory_total=12 * GIB,
            cuda_memory_available=11 * GIB,
        )

    @patch(
        "backend.ai_simulation_core.llm.qwen_model.psutil.cpu_count",
        return_value=16,
    )
    @patch("backend.ai_simulation_core.llm.qwen_model.psutil.virtual_memory")
    @patch("backend.ai_simulation_core.llm.qwen_model.torch.cuda.mem_get_info")
    def test_reads_current_system_and_cuda_resources(
        self,
        mem_get_info: MagicMock,
        virtual_memory: MagicMock,
        _cpu_count: MagicMock,
    ) -> None:
        virtual_memory.return_value = SimpleNamespace(
            total=32 * GIB,
            available=28 * GIB,
        )
        mem_get_info.return_value = (11 * GIB, 12 * GIB)

        resources = get_system_resources(torch.device("cuda"))

        self.assertEqual(resources, self.cuda_resources)
        mem_get_info.assert_called_once_with(0)

    def test_selects_8b_when_cuda_and_system_memory_are_sufficient(self) -> None:
        self.assertEqual(
            select_qwen_model(self.cuda_resources),
            QWEN_8B_MODEL_NAME,
        )

    def test_selects_4b_when_cuda_memory_is_insufficient(self) -> None:
        resources = replace(
            self.cuda_resources,
            cuda_memory_available=8 * GIB,
        )

        self.assertEqual(select_qwen_model(resources), QWEN_4B_MODEL_NAME)

    def test_selects_8b_for_large_mps_unified_memory(self) -> None:
        resources = SystemResources(
            device_type="mps",
            cpu_count=10,
            system_memory_total=64 * GIB,
            system_memory_available=32 * GIB,
        )

        self.assertEqual(select_qwen_model(resources), QWEN_8B_MODEL_NAME)

    def test_selects_8b_for_large_cpu_resources(self) -> None:
        resources = SystemResources(
            device_type="cpu",
            cpu_count=12,
            system_memory_total=64 * GIB,
            system_memory_available=32 * GIB,
        )

        self.assertEqual(select_qwen_model(resources), QWEN_8B_MODEL_NAME)

    def test_cuda_memory_limits_keep_headroom_on_smaller_computer(self) -> None:
        resources = replace(
            self.cuda_resources,
            system_memory_available=16 * GIB,
            cuda_memory_available=8 * GIB,
        )

        self.assertEqual(
            get_cuda_max_memory(resources),
            {0: "7GiB", "cpu": "14GiB"},
        )

    def test_environment_override_takes_priority_over_resource_check(self) -> None:
        resources = SystemResources(
            device_type="cpu",
            cpu_count=2,
            system_memory_total=8 * GIB,
            system_memory_available=4 * GIB,
        )

        self.assertEqual(
            select_qwen_model(
                resources,
                model_name_override=QWEN_8B_MODEL_NAME,
            ),
            QWEN_8B_MODEL_NAME,
        )

    def test_cuda_loads_resource_selected_qwen3_8b_with_cpu_offload(self) -> None:
        tokenizer = MagicMock()
        model = MagicMock()

        with (
            patch.dict("os.environ", {QWEN_MODEL_NAME_ENV: ""}),
            patch.object(LLM, "get_device", return_value=torch.device("cuda")),
            patch(
                "backend.ai_simulation_core.llm.qwen_model.get_system_resources",
                return_value=self.cuda_resources,
            ),
            patch(
                "backend.ai_simulation_core.llm.qwen_model.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_pretrained_tokenizer,
            patch(
                "backend.ai_simulation_core.llm.qwen_model.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as from_pretrained_model,
        ):
            llm = LLM()

        from_pretrained_tokenizer.assert_called_once_with(QWEN_8B_MODEL_NAME)
        from_pretrained_model.assert_called_once_with(
            QWEN_8B_MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: CUDA_MAX_MEMORY, "cpu": CPU_MAX_MEMORY},
            offload_buffers=True,
        )
        model.to.assert_not_called()
        model.eval.assert_called_once_with()
        self.assertEqual(llm.model_name, QWEN_8B_MODEL_NAME)

    def test_structured_generation_is_deterministic(self) -> None:
        llm = LLM.__new__(LLM)
        llm._closed = False
        llm.device = torch.device("cpu")
        llm.tokenizer = _Tokenizer()
        llm.model = _Model()

        response = llm.generate("정책 민원을 JSON으로 생성")

        self.assertEqual(response, "응답")
        self.assertEqual(llm.model.generate_kwargs["max_new_tokens"], 1024)
        self.assertIs(llm.model.generate_kwargs["do_sample"], False)
        self.assertNotIn("temperature", llm.model.generate_kwargs)
        self.assertNotIn("top_p", llm.model.generate_kwargs)


if __name__ == "__main__":
    unittest.main()
