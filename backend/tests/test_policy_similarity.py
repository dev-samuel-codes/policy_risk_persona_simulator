import unittest

from backend.ai_simulation_core.policies.policy_repository import build_direct_policy
from backend.ai_simulation_core.policies.policy_similarity import (
    PolicySimilarityService,
    text_overlap,
    token_dice,
)


class FakeVector(list):
    def tolist(self) -> list[float]:
        return list(self)


class FakeEmbedder:
    def encode(self, *_args, **_kwargs) -> FakeVector:
        return FakeVector([0.1, 0.2, 0.3])


class FakeCollection:
    def __init__(self) -> None:
        self.last_query = None

    def count(self) -> int:
        return 2

    def query(self, **kwargs) -> dict:
        self.last_query = kwargs
        return {
            "ids": [["different-dense", "similar-policy"]],
            "distances": [[0.10, 0.30]],
            "metadatas": [[{}, {}]],
        }


def candidate(
    service_id: str,
    *,
    name: str,
    target: str,
    benefits: str,
) -> dict:
    return {
        "service_id": service_id,
        "policy_name": name,
        "purpose": "",
        "category": "주거",
        "support_type": "현금",
        "target_audience": target,
        "selection_criteria": target,
        "benefits": benefits,
        "application_period": "상시신청",
        "application_method": "온라인 신청",
        "required_documents": "",
        "contact": "",
        "organization": "주거지원청",
        "organization_type": "공공기관",
        "registered_at": "2020-01-01",
        "modified_at": "2026-01-01",
        "source_url": "https://example.test/policy",
        "support_conditions": "",
    }


class PolicySimilarityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = FakeCollection()
        self.service = PolicySimilarityService(
            collection=self.collection,
            embedder=FakeEmbedder(),
            corpus=(
                candidate(
                    "different-dense",
                    name="농업 시설 융자",
                    target="농업 법인",
                    benefits="시설 자금 융자",
                ),
                candidate(
                    "similar-policy",
                    name="청년 주거비 지원",
                    target="만 19세 이상 34세 이하 무주택 청년",
                    benefits="월 20만 원을 12개월간 지원",
                ),
            ),
            manifest={
                "built_at": "2026-08-06T00:00:00+00:00",
                "document_count": 2,
                "embedding_model": "test-model",
                "source": {
                    "provider": "행정안전부",
                    "dataset": "대한민국 공공서비스(혜택) 정보",
                    "api_type": "OpenAPI",
                    "fetched_at": "2026-08-25T00:00:00+00:00",
                },
            },
        )
        self.policy = build_direct_policy(
            {
                "policy_name": "청년 주거 지원",
                "target_audience": "만 19세 이상 34세 이하 무주택 청년",
                "benefits": "월 20만 원 지원",
                "effective_date": "2026-09-01",
            }
        )

    def test_field_reranking_beats_unrelated_dense_candidate(self) -> None:
        result = self.service.search(self.policy, top_k=2, min_score=0)

        self.assertEqual(result["results"][0]["service_id"], "similar-policy")
        self.assertEqual(result["as_of_date"], "2026-09-01")
        self.assertEqual(
            self.collection.last_query["where"],
            {"registered_ordinal": {"$lte": 20260901}},
        )
        self.assertIn(
            "지원내용",
            {reason["field"] for reason in result["results"][0]["match_reasons"]},
        )

    def test_similarity_response_contains_source_fields(self) -> None:
        response = self.service.search(self.policy, top_k=1, min_score=0)
        result = response["results"][0]

        self.assertEqual(result["organization"], "주거지원청")
        self.assertEqual(result["registered_at"], "2020-01-01")
        self.assertEqual(result["benefits"], "월 20만 원을 12개월간 지원")
        self.assertEqual(result["source_url"], "https://example.test/policy")
        self.assertEqual(response["source"]["api_type"], "OpenAPI")
        self.assertEqual(
            response["source"]["fetched_at"],
            "2026-08-25T00:00:00+00:00",
        )

    def test_character_overlap_matches_korean_compound_words(self) -> None:
        query = "소상공인 저금리 경영안정자금"
        candidate_name = "소상공인 정책자금 일반경영안정자금 융자"

        self.assertGreater(
            text_overlap(query, candidate_name),
            token_dice(query, candidate_name),
        )


if __name__ == "__main__":
    unittest.main()
