from pathlib import Path
from typing import Any
import shutil

import pandas as pd
from huggingface_hub import snapshot_download


# 다운로드할 Hugging Face 데이터셋 이름
DATASET_REPO_ID = "nvidia/Nemotron-Personas-Korea"

# 어떤 파일로 받을 지 선택: CSV / JSON / Parquet
REVISION = "refs/convert/parquet"

# 현재 파일 위치 기준으로 backend 폴더 경로 찾기
BACKEND_DIR = Path(__file__).resolve().parents[2]

# 저장 경로 설정
# 최종적으로 backend/data/nometron_persona_korea/*.parquet 형태로 저장
LOCAL_DIR = BACKEND_DIR / "data" / "nometron_persona_korea"

# snapshot_download가 임시로 내려받는 Hugging Face 내부 경로
TEMP_TRAIN_DIR = LOCAL_DIR / "default" / "train"

# 다운로드 완료 여부를 표시하는 marker 파일
COMPLETE_MARKER = LOCAL_DIR / ".download_complete"


# 이미 다운로드 되어있는 지 확인: marker 파일 존재 여부 / parquet 파일 존재 여부 / 각 파일 크기 > 0 인지
def is_already_downloaded() -> bool:
    parquet_files = list(LOCAL_DIR.glob("*.parquet"))

    if not COMPLETE_MARKER.exists():
        return False

    if not parquet_files:
        return False

    return all(file_path.stat().st_size > 0 for file_path in parquet_files)


# Hugging Face에서 받은 default/train/*.parquet 파일을 backend/data/nometron_persona_korea/*.parquet 위치로 이동
def flatten_parquet_files() -> None:
    if not TEMP_TRAIN_DIR.exists():
        raise RuntimeError(
            f"[ERROR] 임시 다운로드 폴더를 찾을 수 없습니다: {TEMP_TRAIN_DIR}"
        )

    parquet_files = sorted(TEMP_TRAIN_DIR.glob("*.parquet"))

    if not parquet_files:
        raise RuntimeError("[ERROR] 이동할 parquet 파일이 없습니다.")

    for source_file in parquet_files:
        target_file = LOCAL_DIR / source_file.name

        if target_file.exists() and target_file.stat().st_size > 0:
            continue

        source_file.replace(target_file)

    default_dir = LOCAL_DIR / "default"

    if default_dir.exists():
        shutil.rmtree(default_dir)


# Nemotron-Personas-Korea 데이터셋의 parquet 파일을 다운로드 또는 파일 존재 시 다운로드 스킵
def download_dataset() -> Path:
    if is_already_downloaded():
        print("[SKIP] 파일이 이미 존재하여 다운로드를 건너뜁니다.")
        print(f"[PATH] {LOCAL_DIR.resolve()}")
        return LOCAL_DIR

    print("[DOWNLOAD] Nemotron-Personas-Korea 데이터를 다운로드 시작합니다.")
    print(f"[TARGET] {LOCAL_DIR.resolve()}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(LOCAL_DIR),
        allow_patterns=["default/train/*.parquet"],
    )

    flatten_parquet_files()

    COMPLETE_MARKER.write_text(
        "Nemotron-Personas-Korea parquet 다운로드 완료\n",
        encoding="utf-8",
    )

    if not is_already_downloaded():
        raise RuntimeError("[ERROR] 데이터셋 다운로드가 정상적으로 완료되지 않았습니다.")

    print("[DONE] 다운로드 완료.")
    print(f"[PATH] {LOCAL_DIR.resolve()}")

    return LOCAL_DIR


def get_local_parquet_files(auto_download: bool = True) -> list[Path]:
    if auto_download:
        dataset_dir = download_dataset()
    else:
        dataset_dir = LOCAL_DIR

    return sorted(dataset_dir.glob("*.parquet"))


# 로컬 parquet 파일에서 사용할 페르소나 가져오기
def get_persona(limit: int = 10) -> list[dict[str, Any]]:
    personas: list[dict[str, Any]] = []

    for parquet_file in get_local_parquet_files():
        dataframe = pd.read_parquet(parquet_file)

        for persona in dataframe.to_dict(orient="records"):
            persona_record = {
                str(key): value
                for key, value in persona.items()
            }
            personas.append(persona_record)

            if len(personas) >= limit:
                return personas

    return personas
