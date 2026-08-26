#!/usr/bin/env python3
"""Build and atomically activate the verified public-FAQ Chroma index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (  # noqa: E402
    FAQ_DATA_DIR,
    build_civil_complaint_search_document,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
    load_civil_complaint_source_metadata,
)
from backend.ai_simulation_core.complaints.civil_complaint_similarity import (  # noqa: E402
    ACTIVE_POINTER_FILENAME,
    ACTIVE_POINTER_SCHEMA_VERSION,
    ACTIVE_RELOAD_STRATEGY,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    MANIFEST_SCHEMA_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "추적 중인 민원 FAQ 정본을 임베딩하고 검증한 뒤 Chroma 인덱스를 "
            "원자적으로 교체합니다."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=FAQ_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        default=os.getenv("CIVIL_COMPLAINT_INDEX_DEVICE", "cuda"),
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--write-batch-size", type=int, default=256)
    return parser.parse_args()


def content_hash(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def collection_metadata(model_name: str) -> dict[str, str]:
    return {
        "hnsw:space": "cosine",
        "schema_version": str(MANIFEST_SCHEMA_VERSION),
        "embedding_model": model_name,
        "source_kind": "public_faq_snapshot",
    }


def _manifest_sha256(index_dir: Path) -> str:
    return hashlib.sha256((index_dir / "manifest.json").read_bytes()).hexdigest()


def _version_name(staging_dir: Path) -> str:
    manifest_bytes = (staging_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    built_at = str(manifest.get("built_at") or "")
    timestamp = re.sub(r"[^0-9]", "", built_at)[:20]
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{timestamp}-{hashlib.sha256(manifest_bytes).hexdigest()[:12]}"


def _write_active_pointer(logical_dir: Path, pointer: dict[str, Any]) -> Path:
    logical_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = logical_dir / ACTIVE_POINTER_FILENAME
    temporary_path = logical_dir / f".{ACTIVE_POINTER_FILENAME}.{uuid4().hex}.tmp"
    payload = (json.dumps(pointer, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with temporary_path.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, pointer_path)
        with suppress(OSError):
            directory_fd = os.open(logical_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
    return pointer_path


def activate_index(staging_dir: Path, target_dir: Path) -> Path:
    """Move staging to an immutable version and atomically swap its pointer.

    Existing files below the logical ``current`` directory are never renamed or
    deleted.  This also migrates a legacy in-place Chroma directory by adding
    only ``current/active.json`` after the new version is durable.
    """

    staging = staging_dir.resolve()
    target = target_dir.resolve()
    if not staging.is_dir():
        raise FileNotFoundError(f"검증된 staging 인덱스가 없습니다: {staging}")
    index_root = target.parent
    versions_root = index_root / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != versions_root.stat().st_dev:
        raise ValueError(
            "staging과 versions는 원자적 rename을 위해 같은 파일시스템이어야 합니다."
        )
    version_name = _version_name(staging)
    version_dir = versions_root / version_name
    if version_dir.exists():
        raise FileExistsError(f"민원 FAQ version이 이미 존재합니다: {version_dir}")

    # The immutable version becomes durable first.  A crash before os.replace
    # only leaves an unreferenced version; the previous live pointer is intact.
    staging.rename(version_dir)
    pointer = {
        "schema_version": ACTIVE_POINTER_SCHEMA_VERSION,
        "active_version": version_name,
        "version_path": os.path.relpath(version_dir, start=target),
        "manifest_sha256": _manifest_sha256(version_dir),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "reload_strategy": ACTIVE_RELOAD_STRATEGY,
        "restart_required": False,
    }
    _write_active_pointer(target, pointer)
    return version_dir


def _validate_staging(
    staging_dir: Path,
    *,
    expected_ids: set[str],
    expected_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    import chromadb

    manifest_path = staging_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in expected_fingerprint.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"staging manifest의 {key} 검증에 실패했습니다.")

    client = chromadb.PersistentClient(path=str(staging_dir))
    collection = client.get_collection(manifest["collection_name"])
    collection_meta = collection.metadata
    if not isinstance(collection_meta, dict):
        raise RuntimeError("staging collection metadata가 없습니다.")
    expected_collection_meta = {
        "hnsw:space": "cosine",
        "schema_version": str(manifest["schema_version"]),
        "embedding_model": manifest["embedding_model"],
        "source_kind": "public_faq_snapshot",
    }
    for key, value in expected_collection_meta.items():
        if collection_meta.get(key) != value:
            raise RuntimeError(
                f"staging collection metadata의 {key} 검증에 실패했습니다."
            )

    snapshot = collection.get(include=["documents", "metadatas", "embeddings"])
    indexed_ids = set(snapshot["ids"])
    if collection.count() != len(expected_ids) or indexed_ids != expected_ids:
        raise RuntimeError(
            "staging 인덱스 검증 실패: canonical case_id와 Chroma ID가 다릅니다."
        )
    embedding_dimension = int(manifest.get("embedding_dimension") or 0)
    if embedding_dimension < 1:
        raise RuntimeError("staging manifest의 embedding_dimension이 없습니다.")
    documents = snapshot.get("documents")
    metadatas = snapshot.get("metadatas")
    embeddings = snapshot.get("embeddings")
    if (
        documents is None
        or metadatas is None
        or embeddings is None
        or len(documents) != len(expected_ids)
        or len(metadatas) != len(expected_ids)
        or len(embeddings) != len(expected_ids)
    ):
        raise RuntimeError("staging collection의 검증 필드 건수가 올바르지 않습니다.")
    for case_id, document, metadata, embedding in zip(
        snapshot["ids"], documents, metadatas, embeddings, strict=True
    ):
        if not isinstance(document, str) or not document:
            raise RuntimeError(f"staging 문서가 비어 있습니다: {case_id}")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"staging metadata가 없습니다: {case_id}")
        if (
            metadata.get("case_id") != case_id
            or metadata.get("source_kind") != "public_faq_snapshot"
        ):
            raise RuntimeError(f"staging provenance metadata가 다릅니다: {case_id}")
        stored_hash = str(metadata.get("content_hash") or "")
        if stored_hash != content_hash(document):
            raise RuntimeError(f"staging content_hash가 다릅니다: {case_id}")
        if len(embedding) != embedding_dimension:
            raise RuntimeError(f"staging embedding dimension이 다릅니다: {case_id}")
    return manifest


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    if args.encode_batch_size < 1 or args.write_batch_size < 1:
        raise ValueError("배치 크기는 1 이상이어야 합니다.")

    data_dir = args.data_dir.resolve()
    target_dir = args.index_dir.resolve()
    index_root = target_dir.parent
    index_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=index_root))

    try:
        corpus = list(load_civil_complaint_corpus(data_dir))
        fingerprint = civil_complaint_source_fingerprint(data_dir)
        case_ids = [record["case_id"] for record in corpus]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("canonical 민원 FAQ에 중복 case_id가 있습니다.")
        if len(corpus) != fingerprint["unique_count"]:
            raise ValueError("canonical 민원 FAQ 건수가 source fingerprint와 다릅니다.")

        print(
            f"민원 FAQ 원본: {fingerprint['raw_record_count']:,}행 / "
            f"고유 {len(corpus):,}건"
        )
        print(f"임시 인덱스: {staging_dir}")
        print(f"임베딩 모델: {args.model} ({args.device})")

        model = SentenceTransformer(args.model, device=args.device)
        model.max_seq_length = 512
        client = chromadb.PersistentClient(path=str(staging_dir))
        collection = client.create_collection(
            name=DEFAULT_COLLECTION_NAME,
            metadata=collection_metadata(args.model),
        )

        for start in range(0, len(corpus), args.write_batch_size):
            batch = corpus[start : start + args.write_batch_size]
            documents = [
                build_civil_complaint_search_document(record) for record in batch
            ]
            embeddings = model.encode(
                documents,
                batch_size=args.encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            collection.add(
                ids=[record["case_id"] for record in batch],
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=[
                    {
                        "case_id": record["case_id"],
                        "organization": record["organization"],
                        "content_hash": content_hash(document),
                        "source_kind": "public_faq_snapshot",
                    }
                    for record, document in zip(batch, documents, strict=True)
                ],
            )
            completed = min(start + len(batch), len(corpus))
            print(f"인덱싱: {completed:,}/{len(corpus):,}")

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "embedding_model": args.model,
            "embedding_dimension": model.get_embedding_dimension(),
            "max_sequence_length": model.max_seq_length,
            **fingerprint,
            "source": load_civil_complaint_source_metadata(data_dir),
            "activation": {
                "kind": "atomic_pointer",
                "logical_path": "data/indexes/civil_complaints/current",
                "pointer_file": ACTIVE_POINTER_FILENAME,
                "reload_strategy": ACTIVE_RELOAD_STRATEGY,
                "restart_required": False,
            },
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Reopen the on-disk staging collection and compare every ID before the
        # live path can be touched.
        _validate_staging(
            staging_dir,
            expected_ids=set(case_ids),
            expected_fingerprint=fingerprint,
        )
        version_dir = activate_index(staging_dir, target_dir)
        manifest["index_path"] = str(version_dir)
        manifest["logical_index_path"] = str(target_dir)
        manifest["active_pointer_path"] = str(target_dir / ACTIVE_POINTER_FILENAME)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def main() -> None:
    build_index(parse_args())


if __name__ == "__main__":
    main()
