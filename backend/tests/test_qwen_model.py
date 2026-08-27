import unittest
from unittest.mock import MagicMock, patch

import torch

from backend.ai_simulation_core.llm.qwen_model import (
    CPU_MAX_MEMORY,
    CUDA_MAX_MEMORY,
    LLM,
    QWEN_MODEL_NAME,
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
    def test_cuda_loads_qwen3_8b_with_cpu_offload(self) -> None:
        tokenizer = MagicMock()
        model = MagicMock()

        with (
            patch.object(LLM, "get_device", return_value=torch.device("cuda")),
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

        from_pretrained_tokenizer.assert_called_once_with(QWEN_MODEL_NAME)
        from_pretrained_model.assert_called_once_with(
            QWEN_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: CUDA_MAX_MEMORY, "cpu": CPU_MAX_MEMORY},
            offload_buffers=True,
        )
        model.to.assert_not_called()
        model.eval.assert_called_once_with()
        self.assertEqual(llm.model_name, "Qwen/Qwen3-8B")

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
