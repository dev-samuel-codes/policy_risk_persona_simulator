# backend/ai_simulation_core/get_nemotron_personas.py

# 네모트론 페르소나 데이터 다운로드

from pathlib import Path
from huggingface_hub import snapshot_download
import shutil


# 다운로드할 Hugging Face 데이터셋 이름
DATASET_REPO_ID = "nvidia/Nemotron-Personas-Korea"

# 어떤 파일로 받을 지 선택: CSV / JSON / Parquet
REVISION = "refs/convert/parquet"

# 현재 파일 위치 기준으로 backend 폴더 경로 찾기
BACKEND_DIR = Path(__file__).resolve().parents[1]

# 저장 경로 설정
# 최종적으로 data/nemotron_personas_korea/*.parquet 형태로 저장
LOCAL_DIR = BACKEND_DIR / "dataset" / "nemotron_personas_korea"

# snapshot_download가 임시로 내려받는 Hugging Face 내부 경로
TEMP_TRAIN_DIR = LOCAL_DIR / "default" / "train"

# 다운로드 완료 여부를 표시하는 marker 파일
COMPLETE_MARKER = LOCAL_DIR / ".download_complete"


# 다운로드 되어있어야 하는 Parquet 목록을 미리 생성
EXPECTED_PARQUET_FILES = [
    LOCAL_DIR / f"{i:04d}.parquet"
    for i in range(9)
]


# 이미 다운로드 되어있는 지 확인:
# marker 파일 존재 여부 / 예상 Parquet 파일 존재 여부 / 각 파일 크기 > 0 인지
def is_already_downloaded() -> bool:
    if not COMPLETE_MARKER.exists():
        return False

    for file_path in EXPECTED_PARQUET_FILES:
        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

    return True


# Hugging Face에서 받은 default/train/*.parquet 파일을
# data/nemotron_personas_korea/*.parquet 위치로 이동
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

        shutil.move(str(source_file), str(target_file))

    # 비어 있는 default/train 폴더 정리
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

    # Hugging Face에서 파일 다운로드
    # 주의: snapshot_download는 저장소 구조를 유지하므로 처음에는 default/train/*.parquet로 저장
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(LOCAL_DIR),
        allow_patterns=[
            "default/train/*.parquet",
        ],
    )

    # default/train/*.parquet -> data/nemotron_personas_korea/*.parquet 로 이동
    flatten_parquet_files()

    # 다운로드 후 파일 검사: 제대로 생겼는지 확인
    missing_files = [
        file_path
        for file_path in EXPECTED_PARQUET_FILES
        if not file_path.exists() or file_path.stat().st_size == 0
    ]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise RuntimeError(
            "[ERROR] 몇몇 데이터가 정확히 다운로드 되지 않았습니다:\n"
            f"{missing_text}"
        )

    COMPLETE_MARKER.write_text(
        "Nemotron-Personas-Korea parquet 다운로드 완료\n",
        encoding="utf-8",
    )

    print("[DONE] 다운로드 완료.")
    print(f"[PATH] {LOCAL_DIR.resolve()}")

    return LOCAL_DIR