# llm 실행

from typing import Any

def run_qwen(prompt: str) -> str:
    """
    나중에 QwenLocalLLM 연결부를 여기에 넣으면 됩니다.
    지금은 테스트용 더미 응답입니다.
    """
    return f"Qwen response for: {prompt}"


def main() -> None:
    result = run_qwen("테스트 프롬프트")
    print(result)


if __name__ == "__main__":
    main()