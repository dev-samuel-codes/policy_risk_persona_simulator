#!/usr/bin/env python3
"""정부24 보조금24 OpenAPI 정책 스냅샷을 갱신한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.policies.gov24_openapi import (  # noqa: E402
    Gov24OpenAPIClient,
    fetch_and_write_policy_snapshot,
    service_key_from_environment,
)
from backend.ai_simulation_core.policies.policy_corpus import (  # noqa: E402
    POLICY_DATA_DIR,
)

DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "행정안전부 대한민국 공공서비스(혜택) 정보 OpenAPI의 목록·상세·"
            "지원조건을 검증한 뒤 정책 데이터 디렉터리를 교체합니다."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=POLICY_DATA_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="OpenAPI 응답을 끝까지 수집·검증하되 파일은 바꾸지 않습니다.",
    )
    return parser.parse_args()


def snapshot_summary(snapshot: dict) -> dict:
    first = next(iter(snapshot.values()))
    return {
        "provider": first["source"]["provider"],
        "dataset": first["source"]["dataset"],
        "api_type": first["source"]["api_type"],
        "fetched_at": first["fetched_at"],
        "resources": {
            filename: payload["count"] for filename, payload in snapshot.items()
        },
    }


def main() -> None:
    args = parse_args()
    service_key = service_key_from_environment(args.env_file)
    if args.dry_run:
        snapshot = Gov24OpenAPIClient(
            service_key=service_key,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        ).fetch_snapshot()
        backup = None
    else:
        snapshot, backup = fetch_and_write_policy_snapshot(
            target_dir=args.data_dir,
            service_key=service_key,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        )

    summary = snapshot_summary(snapshot)
    summary["mode"] = "dry-run" if args.dry_run else "updated"
    summary["data_dir"] = str(args.data_dir.resolve())
    summary["backup_dir"] = str(backup) if backup else None
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.dry_run:
        print(
            "정책 원천이 바뀌었습니다. 다음 명령으로 유사정책 인덱스를 재생성하세요:\n"
            "  python scripts/rag/build_policy_index.py"
        )


if __name__ == "__main__":
    main()
