# 실행 전용 파일

from backend.ai_simulation_core.llm_gateway.models import QwenLocalLLM


def main() -> None:
    llm = QwenLocalLLM()

    while True:
        prompt = input("\n사용자> ").strip()

        if prompt.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        if not prompt:
            continue

        response = llm.chat(prompt)
        print(f"\nQwen> {response}")


if __name__ == "__main__":
    main()