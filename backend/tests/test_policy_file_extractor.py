import json
import unittest
from unittest.mock import patch

from backend.ai_simulation_core.policies.policy_file_extractor import (
    FIELD_LABELS,
    POLICY_FIELD_KEYS,
    SUPPORTED_EXTENSIONS,
    PolicyFileExtractionError,
    _build_extraction_prompt,
    extract_policy_fields,
    extract_text_from_file,
)


class PolicyFileExtractorTest(unittest.TestCase):
    def test_only_verified_extensions_are_supported(self) -> None:
        self.assertEqual(
            SUPPORTED_EXTENSIONS,
            {".pdf", ".docx", ".hwpx", ".txt", ".md"},
        )

    def test_hwp_is_rejected_as_unsupported(self) -> None:
        with self.assertRaisesRegex(
            PolicyFileExtractionError,
            "지원하지 않는 파일 형식",
        ):
            extract_text_from_file("policy.hwp", b"not-an-hwp")

    def test_extraction_prompt_is_policy_only(self) -> None:
        prompt = _build_extraction_prompt("정책 지원 내용")

        self.assertIn("정책 공고문", prompt)
        self.assertNotIn("법령", prompt)

    def test_selection_criteria_is_an_independent_extraction_field(self) -> None:
        prompt = _build_extraction_prompt("소득순으로 우선 선정합니다.")

        self.assertIn("selection_criteria", POLICY_FIELD_KEYS)
        self.assertEqual(FIELD_LABELS["selection_criteria"], "선정기준")
        self.assertIn('"selection_criteria" (선정기준)', prompt)

    @patch(
        "backend.ai_simulation_core.policies.policy_file_extractor.run_llm",
        return_value=json.dumps(
            {
                "policy_name": "청년 지원",
                "selection_criteria": "  소득순 심사  ",
                "exclusion_conditions": None,
            },
            ensure_ascii=False,
        ),
    )
    def test_selection_criteria_is_preserved_and_missing_values_are_blank(
        self,
        _mock_run_llm,
    ) -> None:
        fields = extract_policy_fields("청년 지원 공고")

        self.assertEqual(fields["selection_criteria"], "소득순 심사")
        self.assertEqual(fields["exclusion_conditions"], "")
        self.assertEqual(fields["target_audience"], "")


if __name__ == "__main__":
    unittest.main()
