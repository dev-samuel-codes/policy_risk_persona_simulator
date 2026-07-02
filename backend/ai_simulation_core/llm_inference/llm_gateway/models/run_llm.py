# backend/ai_simulation_core/llm_inference/llm_gateway/models/run_llm.py

from backend.ai_simulation_core.llm_inference.llm_gateway.models.qwen_local import QwenLocalLLM


_llm: QwenLocalLLM | None = None


def get_llm() -> QwenLocalLLM:
    """
    로컬 LLM을 한 번만 로딩합니다.
    main.py에서 persona마다 run_qwen을 호출해도 모델은 재로딩되지 않습니다.
    """
    global _llm

    if _llm is None:
        print("[LLM] Loading local Qwen model...")
        _llm = QwenLocalLLM()
        print("[LLM] Local Qwen model loaded.")

    return _llm


def run_qwen(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(prompt)