"""Canonical civil-complaint FAQ snapshot loading and integrity helpers.

Only the tracked detail snapshot is allowed to supply searchable evidence.  The
``matchedPolicy`` and ``matchedKeyword`` fields were produced by an earlier
collection heuristic and deliberately never enter the canonical records.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAQ_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "faq"
DETAIL_FILENAME = "civil_policy_qna_detail.json"
METADATA_FILENAME = "civil_policy_qna_metadata.json"

HTML_BREAK_PATTERN = re.compile(r"<\s*(?:br|/p|/div|/li)\s*/?\s*>", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Decode stored HTML and return deterministic plain text."""

    text = html.unescape(str(value or ""))
    text = HTML_BREAK_PATTERN.sub("\n", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", html.unescape(text)).strip()


def canonical_json_sha256(payload: object) -> str:
    """Hash JSON values independently from source whitespace or key ordering."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"민원 FAQ 원천 파일을 읽을 수 없습니다: {path}") from error


def _detail_rows(payload: object, *, path: Path) -> list[dict[str, Any]]:
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"민원 FAQ detail 형식이 올바르지 않습니다: {path}")
    return rows


def _canonical_laws(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    laws: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        law = {
            "full_name": normalize_text(item.get("fullName")),
            "name": normalize_text(item.get("lwrdNm")),
            "url": normalize_text(item.get("lwrdUrl")),
        }
        key = (law["full_name"], law["name"], law["url"])
        if not any(key) or key in seen:
            continue
        seen.add(key)
        laws.append(law)
    return laws


def canonicalize_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only identity and the five authoritative evidence fields."""

    faq_no = normalize_text(detail.get("faqNo"))
    if not faq_no:
        raise ValueError("민원 FAQ detail에 faqNo가 없습니다.")

    title = normalize_text(detail.get("qnaTitl"))
    question = normalize_text(detail.get("qstnCntnCl"))
    answer = normalize_text(detail.get("ansCntnCl"))
    organization = normalize_text(detail.get("ancName"))
    if not title or not question or not answer or not organization:
        raise ValueError(
            f"민원 FAQ {faq_no}의 제목·질문·답변·기관 중 비어 있는 값이 있습니다."
        )

    return {
        "case_id": faq_no,
        "title": title,
        "question": question,
        "answer": answer,
        "organization": organization,
        "related_laws": _canonical_laws(detail.get("lawList")),
    }


def load_civil_complaint_corpus(
    data_dir: str | Path = FAQ_DATA_DIR,
) -> tuple[dict[str, Any], ...]:
    """Load one canonical record per faqNo, rejecting inconsistent duplicates."""

    detail_path = Path(data_dir) / DETAIL_FILENAME
    rows = _detail_rows(_read_json(detail_path), path=detail_path)
    raw_detail_by_id: dict[str, str] = {}
    canonical_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        detail = row.get("detail")
        if not isinstance(detail, Mapping):
            raise ValueError("민원 FAQ 행에 detail 객체가 없습니다.")

        row_faq_no = normalize_text(row.get("faqNo"))
        detail_faq_no = normalize_text(detail.get("faqNo"))
        if not row_faq_no or not detail_faq_no or row_faq_no != detail_faq_no:
            raise ValueError(
                "민원 FAQ 행 faqNo와 detail faqNo가 없거나 서로 다릅니다: "
                f"{row_faq_no!r} != {detail_faq_no!r}"
            )

        # Compare the complete detail object, not the old matchedPolicy metadata.
        raw_signature = json.dumps(
            detail,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous_signature = raw_detail_by_id.get(detail_faq_no)
        if previous_signature is not None:
            if previous_signature != raw_signature:
                raise ValueError(
                    f"동일 faqNo의 detail 내용이 서로 다릅니다: {detail_faq_no}"
                )
            continue

        raw_detail_by_id[detail_faq_no] = raw_signature
        canonical_by_id[detail_faq_no] = canonicalize_detail(detail)

    if not canonical_by_id:
        raise ValueError("민원 FAQ detail에 검색할 레코드가 없습니다.")
    return tuple(canonical_by_id[key] for key in sorted(canonical_by_id))


def civil_complaint_source_fingerprint(
    data_dir: str | Path = FAQ_DATA_DIR,
) -> dict[str, Any]:
    """Return hashes and counts that bind an index to both tracked snapshots."""

    data_path = Path(data_dir)
    detail_path = data_path / DETAIL_FILENAME
    metadata_path = data_path / METADATA_FILENAME
    detail_payload = _read_json(detail_path)
    metadata_payload = _read_json(metadata_path)
    rows = _detail_rows(detail_payload, path=detail_path)
    corpus = load_civil_complaint_corpus(data_path)
    return {
        "detail_sha256": canonical_json_sha256(detail_payload),
        "metadata_sha256": canonical_json_sha256(metadata_payload),
        "raw_record_count": len(rows),
        "unique_count": len(corpus),
    }


def build_civil_complaint_search_document(record: Mapping[str, Any]) -> str:
    """Build the sole document embedded for each public FAQ case."""

    laws = record.get("related_laws")
    law_names = []
    if isinstance(laws, list):
        for law in laws:
            if isinstance(law, Mapping):
                name = normalize_text(law.get("full_name") or law.get("name"))
                if name:
                    law_names.append(name)
    lines = [
        f"제목: {normalize_text(record.get('title'))}",
        f"질문: {normalize_text(record.get('question'))}",
        f"답변: {normalize_text(record.get('answer'))}",
        f"기관: {normalize_text(record.get('organization'))}",
    ]
    if law_names:
        lines.append(f"관련 법령: {' | '.join(law_names)}")
    return "\n".join(lines)


def load_civil_complaint_source_metadata(
    data_dir: str | Path = FAQ_DATA_DIR,
) -> dict[str, Any]:
    """Expose stable provenance without leaking collector-local absolute paths."""

    fingerprint = civil_complaint_source_fingerprint(data_dir)
    return {
        "source_kind": "public_faq_snapshot",
        "dataset": "공공 민원 FAQ 공개 스냅샷",
        "detail_file": f"data/raw/faq/{DETAIL_FILENAME}",
        "metadata_file": f"data/raw/faq/{METADATA_FILENAME}",
        "matched_policy_used": False,
        **fingerprint,
    }
