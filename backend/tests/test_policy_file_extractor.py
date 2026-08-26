import unittest

from backend.ai_simulation_core.policies.policy_file_extractor import (
    SUPPORTED_EXTENSIONS,
    PolicyFileExtractionError,
    _build_extraction_prompt,
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


if __name__ == "__main__":
    unittest.main()
