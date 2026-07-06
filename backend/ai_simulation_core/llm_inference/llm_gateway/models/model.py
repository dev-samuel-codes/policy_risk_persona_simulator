# 로컬 LLM 로드 및 응답 생성 모듈

"""
다른 파일에서 

llm = QwenLocalLLM()
anwser = llm.generate(prompt)
print(answer)

를 작성하면 사용가능
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM을 로컬에서 실행하기 위한 클래스
class LLM:

    # 윈도우, 맥에서 mps, cpu, gpu 구분
    def get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # 클래스 초기화: LLM 객체를 만들 때 자동으로 실행
    def __init__(self) -> None:

        self.model_name = "Qwen/Qwen3-1.7B"
        self.device = self.get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 모델 로드
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype = "auto", # float32: 정확도 좋음, 메모리 많이 사용 / float16: 메모리 적게 사용 / bfloat16: 일부 환경에서 효율적
        )

        self.model.to(self.device)
        self.model.eval()

    # 사용자 질문을 받아서 모델 답변을 문자열로 반환
    def generate(self, prompt: str) -> str:

        #메세지 구성
        messages = [
            {
                "role": "system", # 모델의 역할과 행동기준 설정 : content
                "content": "당신은 한국인입니다. 항상 한국말로 답하세요.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # 채팅 템플릿 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # 아직 숫자 토큰으로 변환하지 않고 문자열 상태로 반환
            add_generation_prompt=True, # 모델에게 모델이 답변할 차례라는 신호를 붙임
            enable_thinking=False       # Thinking 내용 생략
        )

        # 토큰화: 문자열을 모델 입력용 텐서로 변환
        model_inputs = self.tokenizer(
            [text], # 문자열을 하나를 리스트로 랩핑
            return_tensors="pt",
        ).to(self.device)

        # 답변 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,     # 새로 생성할 수 있는 최대 토큰 수
                do_sample=True,         # 답변 생성 시 항상 가장 높은 단어를 고를지 또는 확률적으로 샘플링 할지
                temperature=0.7,        # 창의성 또는 랜덤성 조절: 0.7은 균형 잡힌 값, 낮으면 안정적 및 보수적
                top_p=0.9,              # 확률이 높은 후보들을 누적 확률를 선택
            )

        # 입력 프롬프트 부분 제거
        generated_ids = [
            output_ids[len(input_ids):] # 입력 길이 만큼 앞부분 제거
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids,
            )
        ]

        # 토큰을 문자열로 변환
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,   # 특수 토큰 제거
        )[0]

        return response