import unittest
from unittest.mock import patch

from backend.ai_simulation_core import pipeline


class PipelineResultTest(unittest.TestCase):
    def test_citizen_result_keeps_sampled_persona(self) -> None:
        policy = {"상세정보": {"서비스명": "청년 주거 지원"}}
        citizen_persona = {
            "uuid": "citizen-1",
            "occupation": "조사 전문가",
            "age": 43,
            "province": "경기",
            "district": "성남시 분당구",
        }
        civil_servant_persona = {
            "uuid": "official-1",
            "occupation": "일반 행정 공무원",
            "age": 38,
            "province": "전북",
            "district": "전주시",
        }
        citizen_result = {
            "persona_summary": {"이름": "김진훈", "나이": "43"},
            "personality": "현실적인 주거비 부담을 걱정한다.",
            "complaints": [
                {
                    "complaint_text": "지원 연령에서 제외된다.",
                    "dialogue": "저는 지원받을 수 없어서 답답합니다.",
                }
            ],
        }

        with (
            patch.object(
                pipeline,
                "get_citizen_persona",
                return_value=[citizen_persona],
            ),
            patch.object(
                pipeline,
                "get_civil_servant_persona",
                return_value=[civil_servant_persona],
            ),
            patch.object(
                pipeline,
                "run_citizen_simulation",
                return_value=citizen_result,
            ),
            patch.object(
                pipeline,
                "run_civil_servant_simulation",
                return_value="민원 내용을 확인하겠습니다.",
            ),
            patch.object(
                pipeline,
                "add_risk_categories",
                side_effect=lambda results: results,
            ),
            patch.object(pipeline, "load_risk_pack", return_value=[]),
            patch.object(
                pipeline,
                "compute_index",
                return_value={"score": 35.0},
            ),
        ):
            result = pipeline.run_pipeline(policy=policy)

        saved_citizen = result["citizen_results"][0]
        self.assertEqual(saved_citizen["persona"], citizen_persona)
        self.assertEqual(
            saved_citizen["complaints"][0]["dialogue"],
            "저는 지원받을 수 없어서 답답합니다.",
        )
        self.assertEqual(result["risk_score"]["score"], 35.0)
        self.assertEqual(
            result["civil_servant_results"][0]["persona"],
            civil_servant_persona,
        )


if __name__ == "__main__":
    unittest.main()
