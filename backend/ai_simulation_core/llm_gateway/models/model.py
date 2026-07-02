# LLM 공통 인터페이스

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class BaseLLM(ABC):
    """
    모든 LLM 구현체가 따라야 하는 공통 인터페이스.
    """

    model_name: str
    device: str

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
    ) -> str:
        """
        사용자 프롬프트를 받아 모델 응답을 반환
        """
        pass