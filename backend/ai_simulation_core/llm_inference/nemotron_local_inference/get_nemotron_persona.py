
"""
Nemotron-Personas-Korea에서 Persona를 가져옴
limit으로 개수 설정
"""

from itertools import islice
from typing import Any

from datasets import load_dataset

REPO_ID = "nvidia/Nemotron-Personas-Korea"

def get_persona(limit: int = 10) -> list[dict[str, Any]]:
    
    # 데이터셋 불러오기
    dataset = load_dataset(
        REPO_ID,
        split="train",  # train: 학습용 데이터, test: 테스트용 데이터, validation: 검증용 데이터
        streaming=True, # 전제 데이터셋을 다운하지 않고 필요한 데이터만 순차적으로 읽음
    )

    return list(islice(dataset, limit))
