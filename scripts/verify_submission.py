#!/usr/bin/env python3
"""제출본의 제품 런타임 경계와 활성 데이터 무결성을 검사한다."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (  # noqa: E402
    civil_complaint_source_fingerprint,
)

REQUIRED_FILES = (
    "backend/api.py",
    "backend/ai_simulation_core/personas/persona_downloader.py",
    "data/raw/policies/service_list.json",
    "data/raw/policies/service_detail.json",
    "data/raw/policies/support_conditions.json",
    "data/raw/faq/civil_policy_qna_detail.json",
    "data/raw/faq/civil_policy_qna_metadata.json",
    "data/indexes/policies/current/manifest.json",
    "data/indexes/policies/current/chroma.sqlite3",
    "data/indexes/civil_complaints/current/active.json",
    "scripts/data/fetch_gov24_policy_openapi.py",
    "scripts/rag/build_policy_index.py",
    "scripts/rag/build_civil_complaint_index.py",
)

# 제품 제출 범위에서 제거한 연구 산출물과 레거시 경로의 재유입 방지 목록이다.
# 법령 관련 항목도 현행 기능이 아니라 제거 상태를 검증하기 위해 남겨 둔 이름이다.
FORBIDDEN_PATHS = (
    "outputs",
    "experiments",
    "tests",
    "scripts/training",
    "scripts/oneoff",
    "scripts/policy_json_converter.py",
    "scripts/rag/evaluate_policy_similarity.py",
    "policy_extracor",
    "policy_simulator(rag)",
    "law_simulator(rag)",
    "data/training",
    "data/raw/laws",
    "data/raw/reactions",
    "data/raw/petitions",
    "data/raw/qna_calls",
    "data/raw/civil_complaints",
    "data/raw/faq/civil_policy_qna_list.json",
    "design-qa.md",
    "frontend/README.md",
    "frontend/public/icons.svg",
    "frontend/src/components/ChatInputBar.jsx",
    "frontend/src/components/RiskGauge.jsx",
    "frontend/src/components/PersonaSidebar.jsx",
    "frontend/src/components/SimulationLayout.jsx",
    "frontend/src/assets/images/policy-hero.jpg",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON을 읽을 수 없습니다: {relative_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 최상위 값이 객체가 아닙니다: {relative_path}")
    return payload


def has_index_payload(index_dir: Path) -> bool:
    return (index_dir / "chroma.sqlite3").is_file() and any(
        path.is_file() and path.suffix == ".bin" for path in index_dir.rglob("*.bin")
    )


def git_ignored(relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode == 0


def verify() -> dict[str, Any]:
    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative_path).is_file():
            errors.append(f"필수 파일 없음: {relative_path}")

    for relative_path in FORBIDDEN_PATHS:
        if (PROJECT_ROOT / relative_path).exists():
            errors.append(f"연구·레거시 경로 잔존: {relative_path}")

    backup_indexes = sorted((PROJECT_ROOT / "data/indexes").rglob(".backup-*"))
    staging_indexes = sorted((PROJECT_ROOT / "data/indexes").rglob(".staging-*"))
    for path in backup_indexes + staging_indexes:
        errors.append(f"비활성 인덱스 잔존: {path.relative_to(PROJECT_ROOT)}")

    persona_dir = PROJECT_ROOT / "data/raw/personas"
    persona_files = sorted(persona_dir.glob("*.parquet"))
    persona_marker = persona_dir / ".download_complete"
    persona_delivery = "download_on_demand"

    # 페르소나는 Hugging Face에서 최초 요청 시 자동 다운로드되는 런타임
    # 자산이므로 제출본에 포함하지 않아도 된다. 다만 서버에 캐시가 일부만
    # 남아 있는 상태는 정상 제출 상태로 보지 않는다.
    if persona_files or persona_marker.exists():
        persona_delivery = "local_cache"
        if not persona_files or any(path.stat().st_size == 0 for path in persona_files):
            errors.append("로컬 페르소나 Parquet 캐시가 비어 있거나 불완전합니다.")
        if not persona_marker.is_file():
            errors.append("로컬 페르소나 캐시에 다운로드 완료 marker가 없습니다.")

    if (persona_dir / "default").exists():
        errors.append("페르소나 다운로드 임시 디렉터리가 남아 있습니다.")

    downloader_path = (
        PROJECT_ROOT
        / "backend/ai_simulation_core/personas/persona_downloader.py"
    )
    try:
        downloader_source = downloader_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"페르소나 자동 다운로드 코드를 읽을 수 없습니다: {error}")
    else:
        required_downloader_fragments = (
            "nvidia/Nemotron-Personas-Korea",
            "snapshot_download(",
            'allow_patterns=["default/train/*.parquet"]',
            "get_local_parquet_files(auto_download: bool = True)",
        )
        for fragment in required_downloader_fragments:
            if fragment not in downloader_source:
                errors.append(
                    "페르소나 자동 다운로드 계약이 누락되었습니다: "
                    f"{fragment}"
                )

    try:
        policy_manifest = load_json("data/indexes/policies/current/manifest.json")
        source_hashes = policy_manifest.get("source_hashes")
        if not isinstance(source_hashes, dict):
            errors.append("정책 인덱스 source_hashes가 없습니다.")
        else:
            for filename, expected_hash in source_hashes.items():
                source_path = PROJECT_ROOT / "data/raw/policies" / str(filename)
                if not source_path.is_file() or sha256(source_path) != expected_hash:
                    errors.append(f"정책 원천 해시 불일치: {filename}")
        document_count = policy_manifest.get("document_count")
        if not isinstance(document_count, int) or document_count < 1:
            errors.append("정책 인덱스 문서 수가 유효하지 않습니다.")
        if not has_index_payload(PROJECT_ROOT / "data/indexes/policies/current"):
            errors.append("정책 인덱스 Chroma/HNSW 본체가 없습니다.")
    except ValueError as error:
        errors.append(str(error))

    try:
        pointer = load_json("data/indexes/civil_complaints/current/active.json")
        current_dir = PROJECT_ROOT / "data/indexes/civil_complaints/current"
        civil_root = current_dir.parent.resolve()
        relative_version = pointer.get("version_path")
        if not isinstance(relative_version, str) or not relative_version:
            raise ValueError("민원 인덱스 활성 포인터의 version_path가 없습니다.")
        active_dir = (current_dir / relative_version).resolve()
        if civil_root not in active_dir.parents or not active_dir.is_dir():
            raise ValueError("민원 인덱스 활성 버전 경로가 유효하지 않습니다.")
        manifest_path = active_dir / "manifest.json"
        expected_manifest_hash = pointer.get("manifest_sha256")
        if not manifest_path.is_file() or sha256(manifest_path) != expected_manifest_hash:
            errors.append("민원 인덱스 활성 manifest 해시가 일치하지 않습니다.")
        if not has_index_payload(active_dir):
            errors.append("민원 인덱스 Chroma/HNSW 본체가 없습니다.")

        civil_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = civil_manifest.get("source")
        if not isinstance(source, dict):
            errors.append("민원 인덱스 source 정보가 없습니다.")
        else:
            fingerprint = civil_complaint_source_fingerprint(
                PROJECT_ROOT / "data/raw/faq"
            )
            for manifest_key, expected_value in fingerprint.items():
                if source.get(manifest_key) != expected_value:
                    errors.append(f"민원 원천 fingerprint 불일치: {manifest_key}")
            for manifest_key in ("detail_file", "metadata_file"):
                source_path = source.get(manifest_key)
                if not isinstance(source_path, str) or Path(source_path).is_absolute():
                    errors.append(f"민원 원천 경로가 상대경로가 아님: {manifest_key}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
        active_dir = None

    legacy_current = sorted(
        path.name
        for path in (PROJECT_ROOT / "data/indexes/civil_complaints/current").iterdir()
        if path.name != "active.json"
    )
    if legacy_current:
        errors.append(f"민원 current 레거시 파일 잔존: {', '.join(legacy_current)}")

    index_tracking_targets = [
        "data/indexes/policies/current/manifest.json",
        "data/indexes/civil_complaints/current/active.json",
    ]
    if active_dir is not None:
        index_tracking_targets.append(str(active_dir.relative_to(PROJECT_ROOT) / "manifest.json"))
    for relative_path in index_tracking_targets:
        if git_ignored(relative_path):
            errors.append(f"제출 인덱스가 .gitignore에 걸림: {relative_path}")

    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        env_names = {
            line.split("=", 1)[0].strip()
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
        }
        if "LAW_API_OC" in env_names:
            errors.append("사용하지 않는 LAW_API_OC 환경값이 남아 있습니다.")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "summary": {
            "persona_parquet_files": len(persona_files),
            "persona_delivery": persona_delivery,
            "policy_documents": (
                policy_manifest.get("document_count")
                if "policy_manifest" in locals()
                else None
            ),
            "active_civil_index": (
                str(active_dir.relative_to(PROJECT_ROOT))
                if active_dir is not None
                else None
            ),
            "forbidden_paths_present": sum(
                (PROJECT_ROOT / path).exists() for path in FORBIDDEN_PATHS
            ),
        },
    }


def main() -> None:
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
