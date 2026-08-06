import unittest

from backend.ai_simulation_core.policies.policy_corpus import (
    build_policy_search_document,
    normalize_date,
    normalize_policy_record,
)


class PolicyCorpusTest(unittest.TestCase):
    def test_normalize_policy_record_uses_stable_source_fields(self) -> None:
        policy = normalize_policy_record(
            {
                "서비스ID": " service-1 ",
                "서비스명": "목록 정책명",
                "등록일시": "20200102123456",
                "수정일시": "20260101",
                "서비스분야": "주거",
                "소관기관명": "목록 기관",
            },
            {
                "서비스ID": "service-1",
                "서비스명": "청년 주거 지원",
                "지원대상": "만 19세 이상 청년",
                "지원내용": "월 20만 원 지원",
                "소관기관명": "주거지원청",
            },
            {"서비스ID": "service-1", "JA0101": "Y"},
        )

        self.assertEqual(policy["service_id"], "service-1")
        self.assertEqual(policy["policy_name"], "청년 주거 지원")
        self.assertEqual(policy["benefits"], "월 20만 원 지원")
        self.assertEqual(policy["registered_at"], "2020-01-02")
        self.assertEqual(policy["organization"], "주거지원청")
        self.assertIn("JA0101: Y", policy["support_conditions"])

    def test_search_document_contains_support_content(self) -> None:
        document = build_policy_search_document(
            {
                "policy_name": "청년 주거 지원",
                "target_audience": "무주택 청년",
                "benefits": "월 20만 원 지원",
            }
        )

        self.assertIn("정책명: 청년 주거 지원", document)
        self.assertIn("지원대상: 무주택 청년", document)
        self.assertIn("지원내용: 월 20만 원 지원", document)

    def test_normalize_date_rejects_unknown_text(self) -> None:
        self.assertEqual(normalize_date("20260805112233"), "2026-08-05")
        self.assertEqual(normalize_date("상시신청"), "")


if __name__ == "__main__":
    unittest.main()
