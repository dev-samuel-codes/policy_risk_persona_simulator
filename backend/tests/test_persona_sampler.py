import unittest

from backend.ai_simulation_core.personas.persona_sampler import (
    is_active_civil_servant_occupation,
)


class PersonaSamplerTest(unittest.TestCase):
    def test_current_official_occupations_are_allowed(self) -> None:
        self.assertTrue(is_active_civil_servant_occupation("중앙정부 고위 공무원"))
        self.assertTrue(is_active_civil_servant_occupation("지방정부 고위 공무원"))

    def test_inactive_official_occupations_are_rejected(self) -> None:
        for occupation in (
            "전직 중앙정부 고위 공무원, 현재 구직중",
            "퇴직 지방정부 공무원",
            "은퇴 공무원",
            "공무원 구직자",
        ):
            with self.subTest(occupation=occupation):
                self.assertFalse(is_active_civil_servant_occupation(occupation))

    def test_non_official_occupation_is_rejected(self) -> None:
        self.assertFalse(is_active_civil_servant_occupation("행정 연구원"))


if __name__ == "__main__":
    unittest.main()
