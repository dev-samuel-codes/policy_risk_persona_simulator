import json
import tempfile
import unittest
from pathlib import Path

from backend.ai_simulation_core.complaints.civil_complaint_corpus import (
    FAQ_DATA_DIR,
    canonical_json_sha256,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
)


def _detail(
    faq_no: str = "case-1",
    *,
    title: str = "청년 월세 지원 자격",
    question: str = "만 19세 이상 34세 이하만 가능한가요?",
    answer: str = "만 19세 이상 34세 이하가 대상입니다.",
    organization: str = "경기도 평택시",
) -> dict:
    return {
        "faqNo": faq_no,
        "dutySctnNm": "tqapttn",
        "qnaTitl": title,
        "qstnCntnCl": question,
        "ansCntnCl": answer,
        "ancName": organization,
        "deptName": "복지정책과",
        "regDate": "20260825",
        "ancCode": "123",
        "deptCode": "456",
        "lawList": [
            {
                "fullName": "주거기본법 / 제1조(목적)",
                "lwrdNm": "주거기본법",
                "lwrdUrl": "https://law.go.kr/example",
            }
        ],
        "subjList": [],
    }


def _row(detail: dict, *, matched_policy_name: str = "오래된 휴리스틱") -> dict:
    return {
        "faqNo": detail["faqNo"],
        "list": {
            "faqNo": detail["faqNo"],
            "title": detail["qnaTitl"],
            "matchedPolicy": {"서비스명": matched_policy_name},
        },
        "detail": detail,
        "matchedPolicy": {"서비스명": matched_policy_name},
        "matchedKeyword": "월세",
    }


def _write_snapshot(
    directory: Path, rows: list[dict], metadata: dict | None = None
) -> None:
    (directory / "civil_policy_qna_detail.json").write_text(
        json.dumps({"data": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "civil_policy_qna_metadata.json").write_text(
        json.dumps(metadata or {"savedDetailCount": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )


class CivilComplaintCorpusTest(unittest.TestCase):
    def test_tracked_snapshot_deduplicates_to_authoritative_unique_count(self) -> None:
        corpus = load_civil_complaint_corpus(FAQ_DATA_DIR)
        fingerprint = civil_complaint_source_fingerprint(FAQ_DATA_DIR)

        self.assertEqual(fingerprint["raw_record_count"], 2168)
        self.assertEqual(fingerprint["unique_count"], 1344)
        self.assertEqual(len(corpus), 1344)
        self.assertEqual(len({record["case_id"] for record in corpus}), 1344)
        self.assertEqual(
            set(corpus[0]),
            {
                "case_id",
                "title",
                "question",
                "answer",
                "organization",
                "related_laws",
            },
        )

        serialized = json.dumps(corpus, ensure_ascii=False)
        self.assertNotIn("matchedPolicy", serialized)
        self.assertNotIn("matchedKeyword", serialized)

    def test_canonicalizes_html_and_uses_detail_evidence_only(self) -> None:
        detail = _detail(
            title="청년&nbsp;월세",
            question="<strong>신청</strong><br /> 가능한가요?",
            answer="만 19세 이상&nbsp;34세 이하입니다.",
        )
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_snapshot(data_dir, [_row(detail)])
            record = load_civil_complaint_corpus(data_dir)[0]

        self.assertEqual(record["title"], "청년 월세")
        self.assertEqual(record["question"], "신청 가능한가요?")
        self.assertEqual(record["answer"], "만 19세 이상 34세 이하입니다.")
        self.assertEqual(record["related_laws"][0]["name"], "주거기본법")

    def test_identical_detail_duplicates_merge_even_if_matched_policy_differs(
        self,
    ) -> None:
        detail = _detail()
        rows = [
            _row(detail, matched_policy_name="정책 A"),
            _row(dict(detail), matched_policy_name="정책 B"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_snapshot(data_dir, rows)
            corpus = load_civil_complaint_corpus(data_dir)
            fingerprint = civil_complaint_source_fingerprint(data_dir)

        self.assertEqual(len(corpus), 1)
        self.assertEqual(fingerprint["raw_record_count"], 2)
        self.assertEqual(fingerprint["unique_count"], 1)

    def test_different_detail_for_same_faq_no_fails_closed(self) -> None:
        first = _detail()
        second = _detail(question="서로 다른 질문입니다.")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_snapshot(data_dir, [_row(first), _row(second)])
            with self.assertRaisesRegex(ValueError, "detail 내용이 서로 다릅니다"):
                load_civil_complaint_corpus(data_dir)

    def test_canonical_json_hash_ignores_key_order_and_whitespace(self) -> None:
        left = {"data": [{"b": 2, "a": "한글"}]}
        right = {"data": [{"a": "한글", "b": 2}]}
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))


if __name__ == "__main__":
    unittest.main()
