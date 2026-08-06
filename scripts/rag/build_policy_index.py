#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.policies.policy_corpus import (  # noqa: E402
    POLICY_DATA_DIR,
    build_policy_search_document,
    date_ordinal,
    load_policy_corpus,
    source_hashes,
)

COLLECTION_NAME = "policy_similarity"
DEFAULT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "policies" / "current"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기존 정책 전체를 임베딩해 안전하게 Chroma 인덱스를 구축합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=POLICY_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=os.getenv("POLICY_INDEX_DEVICE", "cuda"))
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--write-batch-size", type=int, default=256)
    return parser.parse_args()


def content_hash(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def collection_metadata(model_name: str) -> dict[str, str]:
    return {
        "hnsw:space": "cosine",
        "schema_version": "1",
        "embedding_model": model_name,
    }


def activate_index(staging_dir: Path, target_dir: Path) -> Path | None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = None
    if target_dir.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = target_dir.parent / f".backup-{timestamp}"
        if backup_dir.exists():
            raise FileExistsError(f"백업 경로가 이미 존재합니다: {backup_dir}")
        target_dir.rename(backup_dir)

    try:
        staging_dir.rename(target_dir)
    except Exception:
        if backup_dir is not None and not target_dir.exists():
            backup_dir.rename(target_dir)
        raise
    return backup_dir


def build_index(args: argparse.Namespace) -> dict:
    import chromadb
    from sentence_transformers import SentenceTransformer

    data_dir = args.data_dir.resolve()
    target_dir = args.index_dir.resolve()
    index_root = target_dir.parent
    index_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=index_root))

    policies = list(load_policy_corpus(data_dir))
    service_ids = [policy["service_id"] for policy in policies]
    if len(service_ids) != len(set(service_ids)):
        raise ValueError("정책 코퍼스에 중복 서비스ID가 있습니다.")
    if any(date_ordinal(policy["registered_at"]) == 0 for policy in policies):
        raise ValueError("등록일시가 없는 정책은 과거 정책 필터에 사용할 수 없습니다.")

    print(f"정책 원본: {len(policies):,}건")
    print(f"임시 인덱스: {staging_dir}")
    print(f"임베딩 모델: {args.model} ({args.device})")

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = 512
    client = chromadb.PersistentClient(path=str(staging_dir))
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=collection_metadata(args.model),
    )

    for start in range(0, len(policies), args.write_batch_size):
        batch = policies[start : start + args.write_batch_size]
        documents = [build_policy_search_document(policy) for policy in batch]
        embeddings = model.encode(
            documents,
            batch_size=args.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        metadatas = []
        for policy, document in zip(batch, documents, strict=True):
            metadatas.append(
                {
                    "service_id": policy["service_id"],
                    "policy_name": policy["policy_name"],
                    "organization": policy["organization"],
                    "category": policy["category"],
                    "registered_at": policy["registered_at"],
                    "registered_ordinal": date_ordinal(policy["registered_at"]),
                    "modified_at": policy["modified_at"],
                    "content_hash": content_hash(document),
                }
            )
        collection.add(
            ids=[policy["service_id"] for policy in batch],
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        completed = min(start + len(batch), len(policies))
        print(f"인덱싱: {completed:,}/{len(policies):,}")

    indexed_ids = collection.get(include=[])["ids"]
    if collection.count() != len(policies) or set(indexed_ids) != set(service_ids):
        raise RuntimeError(
            "인덱스 검증 실패: 원본 서비스ID와 Chroma 서비스ID가 일치하지 않습니다."
        )

    manifest = {
        "schema_version": 1,
        "collection_name": COLLECTION_NAME,
        "embedding_model": args.model,
        "embedding_dimension": model.get_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "document_count": len(policies),
        "unique_service_id_count": len(set(service_ids)),
        "source_hashes": source_hashes(data_dir),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    backup_dir = activate_index(staging_dir, target_dir)
    manifest["index_path"] = str(target_dir)
    manifest["backup_path"] = str(backup_dir) if backup_dir else None
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.write_batch_size < 1:
        raise ValueError("배치 크기는 1 이상이어야 합니다.")
    build_index(args)


if __name__ == "__main__":
    main()
