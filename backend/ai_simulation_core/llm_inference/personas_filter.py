# 원하는 페르소나 추출 파일

from pathlib import Path
from typing import Any
import random

import pandas as pd
import pyarrow.parquet as pq

from backend.ai_simulation_core.llm_inference.get_nemotron_personas import (
    get_local_parquet_files,
)

# 페르소나 추출 시 사용할 컬럼
PERSONA_COLUMNS = [
    "uuid",
    "occupation",
    "sex",
    "age",
    "province",
    "district",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "persona",
]

# pandas / numpy 값을 일반 Python 값으로 변환
def clean_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        return value.item()

    return value


# parquet 파일에서 실제 존재하는 컬럼만 선택
def get_available_columns(parquet_file: Path) -> list[str]:
    parquet_schema = pq.ParquetFile(parquet_file).schema
    available_column_names = set(parquet_schema.names)

    return [
        column
        for column in PERSONA_COLUMNS
        if column in available_column_names
    ]


# parquet 한 줄을 dict 형태의 페르소나로 변환
def row_to_persona(row: pd.Series) -> dict:
    persona = {}

    for column in PERSONA_COLUMNS:
        if column in row.index:
            persona[column] = clean_value(row.get(column))

    return persona


# 원하는 직업 키워드가 포함된 페르소나를 limit 개수만큼 가져오기
def get_persona(
    limit: int = 3,
    keyword: str = "공무원",
    auto_download: bool = True,
) -> list[dict]:
    if limit <= 0:
        raise ValueError("[ERROR] limit은 1 이상이어야 합니다.")

    parquet_files = get_local_parquet_files(auto_download=auto_download)

    if not parquet_files:
        raise FileNotFoundError(
            "[ERROR] parquet 파일을 찾을 수 없습니다. 먼저 download_dataset()을 실행하세요."
        )

    personas = []

    # 매번 같은 파일 순서만 보지 않도록 섞기
    random.shuffle(parquet_files)

    for parquet_file in parquet_files:
        available_columns = get_available_columns(parquet_file)

        if "occupation" not in available_columns:
            continue

        df = pd.read_parquet(
            parquet_file,
            columns=available_columns,
        )

        matched_df = df[
            df["occupation"]
            .fillna("")
            .astype(str)
            .str.contains(keyword, regex=False)
        ]

        if matched_df.empty:
            continue

        # 같은 파일 안에서도 랜덤하게 뽑기
        matched_df = matched_df.sample(frac=1)

        for _, row in matched_df.iterrows():
            personas.append(row_to_persona(row))

            if len(personas) >= limit:
                return personas

    raise ValueError(
        f"[ERROR] '{keyword}' 키워드가 포함된 페르소나를 {limit}개 찾지 못했습니다."
    )


if __name__ == "__main__":
    sample_personas = get_persona(limit=3, keyword="공무원")

    for index, persona in enumerate(sample_personas, start=1):
        print(f"\n===== PERSONA {index} =====")
        print(persona)
