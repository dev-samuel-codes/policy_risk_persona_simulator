#!/usr/bin/env python3
"""E06: E01-style binary KcELECTRA training with original-law features.

One model is fitted on the group-safe train 80%, dev 10% is used for epoch
selection and early stopping, and test 10% is evaluated once after training.
The target is 긍정->찬성 / 부정->반대; neutral rows are excluded. Experiment
Markdown is never edited automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import train_youtube_stance_roberta as core
import train_youtube_stance_roberta_cv as cv
from train_youtube_stance_kcelectra_binary_law_cv import configure_binary_task


def configure_original_law_features(
    law_body_dir: Path,
    top_sections: int,
    max_chars: int,
) -> None:
    original_loader = core.load_silver_data

    def load_with_original_law_context(
        path: Path,
        include_needs_review: bool,
        excluded_video_ids: set[str],
    ) -> tuple[Any, dict[str, Any]]:
        frame, audit = original_loader(
            path,
            include_needs_review,
            excluded_video_ids,
        )
        frame, feature_audit = cv.apply_law_original_context(
            frame,
            law_body_dir,
            top_sections,
            max_chars,
        )
        audit["feature_context"] = feature_audit
        return frame, audit

    core.load_silver_data = load_with_original_law_context


def main() -> None:
    configure_binary_task()
    args = core.parse_args()

    project_dir = Path(
        os.environ.get(
            "PROJECT_DIR",
            "/workspace/samuel/policy_risk_persona_simulator",
        )
    )
    law_body_dir = Path(
        os.environ.get("LAW_BODY_DIR", str(project_dir / "data/raw/laws/bodies"))
    )
    top_sections = int(os.environ.get("LAW_ORIGINAL_TOP_SECTIONS", "3"))
    max_chars = int(os.environ.get("LAW_ORIGINAL_MAX_CHARS", "600"))
    args.context_mode = "law_original"
    args.law_body_dir = law_body_dir
    args.law_original_top_sections = top_sections
    args.law_original_max_chars = max_chars
    configure_original_law_features(law_body_dir, top_sections, max_chars)

    args.run_id = args.run_id or core.make_run_id(args.seed)
    started_at_kst = core.now_kst().isoformat(timespec="seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_summary = core.run_experiment(args, started_at_kst)
    except Exception as error:
        print(
            f"Experiment {args.run_id} failed; run_summary.json was not written: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as output:
        json.dump(
            run_summary,
            output,
            ensure_ascii=False,
            indent=2,
            default=core.json_ready,
        )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, default=core.json_ready))


if __name__ == "__main__":
    main()
