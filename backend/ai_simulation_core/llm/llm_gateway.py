# Qwen 로컬 LLM 단일 로딩 및 호출 모듈

"""
로컬 llm을 한 번만 로딩
"""

from backend.ai_simulation_core.llm.qwen_model import LLM


_llm: LLM | None = (
    None  # 로드된 LLM 객체를 저장해두는 전역변수 / 로드 전 처음 상태: None
)


def get_llm() -> LLM:
    global _llm  # 함수 밖 전역변수 _llm 을 수정

    if _llm is None:
        print("[LLM] 모델 가져오는 중")
        _llm = LLM()  # 모델 로딩
        print("[LLM] 모델 가져오기 완료")

    return _llm


def run_llm(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(prompt)
