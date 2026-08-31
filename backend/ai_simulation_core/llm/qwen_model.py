"""
로컬 Qwen 모델을 로드하고 응답을 생성한다.

사용 예:

    from backend.ai_simulation_core.llm.qwen_model import LLM

    prompt = "정책에 대한 시민 반응을 생성해 주세요."
    llm = LLM()
    try:
        answer = llm.generate(prompt)
        print(answer)
    finally:
        llm.close()
"""

import gc
import os
from dataclasses import dataclass
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GIB = 1024**3
QWEN_4B_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_8B_MODEL_NAME = "Qwen/Qwen3-8B"
QWEN_MODEL_NAME_ENV = "QWEN_MODEL_NAME"

# 자동 전환은 모델을 올리기 직전의 가용 CUDA VRAM만 사용한다.
# RAM과 CPU 정보는 로그 및 오프로딩 상한 계산에만 사용하고 모델 크기
# 결정에는 반영하지 않는다.
QWEN_8B_MIN_AVAILABLE_VRAM = 16 * GIB

CUDA_MEMORY_RESERVE = 1 * GIB
CPU_MEMORY_RESERVE = 2 * GIB
CUDA_MAX_MEMORY_GIB = 9
CPU_MAX_MEMORY_GIB = 22
CUDA_MAX_MEMORY = f"{CUDA_MAX_MEMORY_GIB}GiB"
CPU_MAX_MEMORY = f"{CPU_MAX_MEMORY_GIB}GiB"


@dataclass(frozen=True)
class SystemResources:
    """모델 로딩 직전에 확인한 자원과 오프로딩 계산용 메모리 정보."""

    device_type: str
    cpu_count: int
    system_memory_total: int
    system_memory_available: int
    cuda_memory_total: int | None = None
    cuda_memory_available: int | None = None


def get_system_resources(device: torch.device) -> SystemResources:
    """운영체제와 CUDA API에서 현재 사용 가능한 자원을 읽는다."""
    system_memory = psutil.virtual_memory()
    cuda_memory_available: int | None = None
    cuda_memory_total: int | None = None

    if device.type == "cuda":
        # CUDA가 실제로 사용할 수 있는 현재 여유 VRAM을 읽어 다른 작업이
        # 점유 중인 메모리까지 모델 선택에 반영한다.
        cuda_memory_available, cuda_memory_total = torch.cuda.mem_get_info(
            device.index or 0
        )

    return SystemResources(
        device_type=device.type,
        cpu_count=psutil.cpu_count(logical=True) or 1,
        system_memory_total=system_memory.total,
        system_memory_available=system_memory.available,
        cuda_memory_total=cuda_memory_total,
        cuda_memory_available=cuda_memory_available,
    )


def select_qwen_model(
    resources: SystemResources,
    *,
    model_name_override: str | None = None,
) -> str:
    """가용 CUDA VRAM만으로 Qwen 4B 또는 8B를 선택한다."""
    normalized_override = (model_name_override or "").strip()
    if normalized_override:
        # 재현이 필요한 실행에서는 자동 판단보다 명시적 환경변수를 우선한다.
        return normalized_override

    if (
        resources.device_type == "cuda"
        and resources.cuda_memory_available is not None
        and resources.cuda_memory_available >= QWEN_8B_MIN_AVAILABLE_VRAM
    ):
        return QWEN_8B_MODEL_NAME

    return QWEN_4B_MODEL_NAME


def _format_gib(byte_count: int | None) -> str:
    if byte_count is None:
        return "없음"
    return f"{byte_count / GIB:.1f}GiB"


def _memory_limit(
    available_memory: int,
    *,
    reserve_memory: int,
    maximum_gib: int,
) -> str:
    usable_memory = max(GIB, available_memory - reserve_memory)
    return f"{min(maximum_gib, usable_memory // GIB)}GiB"


def get_cuda_max_memory(resources: SystemResources) -> dict[int | str, str]:
    """현재 여유 메모리를 넘지 않는 CUDA/CPU 배치 상한을 만든다."""
    cuda_memory_available = resources.cuda_memory_available or GIB
    return {
        0: _memory_limit(
            cuda_memory_available,
            reserve_memory=CUDA_MEMORY_RESERVE,
            maximum_gib=CUDA_MAX_MEMORY_GIB,
        ),
        "cpu": _memory_limit(
            resources.system_memory_available,
            reserve_memory=CPU_MEMORY_RESERVE,
            maximum_gib=CPU_MAX_MEMORY_GIB,
        ),
    }


class LLM:
    def get_device(self) -> torch.device:
        """사용 가능한 가속기를 MPS, CUDA, CPU 순으로 선택한다."""
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def __init__(self) -> None:
        self.device = self.get_device()
        self.resources = get_system_resources(self.device)
        self.model_name = select_qwen_model(
            self.resources,
            model_name_override=os.getenv(QWEN_MODEL_NAME_ENV),
        )

        print(
            "[LLM] 시스템 자원 확인: "
            f"장치={self.resources.device_type}, "
            f"CPU={self.resources.cpu_count}코어, "
            "가용 RAM="
            f"{_format_gib(self.resources.system_memory_available)}, "
            "가용 VRAM="
            f"{_format_gib(self.resources.cuda_memory_available)}"
        )
        print(f"[LLM] 가용 VRAM 기준 모델 선택: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # CUDA 경로에서는 BF16과 메모리 상한을 사용해 선택된 Qwen 모델의
        # 레이어와 버퍼를 GPU와 CPU에 자동 배치한다.
        model_kwargs: dict[str, Any] = {"dtype": "auto"}
        if self.device.type == "cuda":
            model_kwargs.update(
                {
                    "dtype": torch.bfloat16,
                    "device_map": "auto",
                    # 감지된 가용량보다 작은 상한만 Accelerate에 전달해
                    # 모델 선택 이후의 레이어 배치에서도 OOM 여지를 줄인다.
                    "max_memory": get_cuda_max_memory(self.resources),
                    "offload_buffers": True,
                }
            )

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self._closed = False

    def generate(self, prompt: str) -> str:
        """프롬프트에 대한 모델 응답을 문자열로 생성한다."""
        if self._closed:
            raise RuntimeError("종료된 LLM은 다시 사용할 수 없습니다.")

        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 아직 숫자 토큰으로 변환하지 않고 문자열 상태로 반환
            add_generation_prompt=True,  # 모델에게 모델이 답변할 차례라는 신호를 붙임
            enable_thinking=False,  # 추론 과정 출력을 비활성화
        )

        # 채팅 템플릿 문자열을 모델 입력 텐서로 변환한다.
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,  # 새로 생성할 수 있는 최대 토큰 수
                # 사실 검증이 필요한 구조화 JSON이므로 확률 샘플링을 사용하지 않는다.
                do_sample=False,
            )

        # 생성 결과에 포함된 입력 토큰을 잘라 새 응답만 남긴다.
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids,
            )
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,  # 특수 토큰 제거
        )[0]

        return response

    def close(self) -> None:
        """모델 참조와 가속기 캐시를 해제한다."""
        if self._closed:
            return

        self._closed = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        self.model = None
        self.tokenizer = None

        # PyTorch 텐서의 마지막 참조를 정리한 뒤 캐시된 VRAM도 반환한다.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
