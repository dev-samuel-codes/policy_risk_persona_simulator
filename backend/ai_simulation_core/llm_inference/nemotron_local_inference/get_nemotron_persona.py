from itertools import islice
from typing import Any

from datasets import load_dataset


REPO_ID = "nvidia/Nemotron-Personas-Korea"


def get_persona_samples(limit: int = 10) -> list[dict[str, Any]]:
    """
    Nemotron-Personas-Korea에서 샘플 persona를 limit개만 가져옵니다.
    전체 데이터셋을 다운로드하지 않고 streaming 방식으로 읽습니다.
    """
    dataset = load_dataset(
        REPO_ID,
        split="train",
        streaming=True,
    )

    return list(islice(dataset, limit))


if __name__ == "__main__":
    samples = get_persona_samples(limit=5)

    for i, sample in enumerate(samples, start=1):
        print(f"\n===== SAMPLE {i} =====")
        print(sample)