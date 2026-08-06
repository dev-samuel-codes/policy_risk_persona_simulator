#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_simulation_core.policies.policy_corpus import (  # noqa: E402
    load_policy_corpus,
)
from backend.ai_simulation_core.policies.policy_similarity import (  # noqa: E402
    PolicySimilarityService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="동일 정책명 교차기관 사례로 유사 정책 Recall@K를 평가합니다."
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def as_pipeline_policy(policy: dict) -> dict:
    return {
        "입력출처": "평가셋",
        "목록정보": {
            "서비스ID": policy["service_id"],
            "서비스명": policy["policy_name"],
            "서비스분야": policy["category"],
            "지원유형": policy["support_type"],
        },
        "상세정보": {
            "서비스ID": policy["service_id"],
            "서비스명": policy["policy_name"],
            "서비스목적": policy["purpose"],
            "지원대상": policy["target_audience"],
            "선정기준": policy["selection_criteria"],
            "지원내용": policy["benefits"],
            "신청기한": policy["application_period"],
            "신청방법": policy["application_method"],
        },
        "지원조건": {},
    }


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[index]


def main() -> None:
    args = parse_args()
    if args.limit < 1 or not 1 <= args.top_k <= 10:
        raise ValueError("limit은 1 이상, top-k는 1에서 10 사이여야 합니다.")

    corpus = load_policy_corpus()
    groups: dict[str, list[dict]] = defaultdict(list)
    for policy in corpus:
        groups[policy["policy_name"]].append(policy)

    duplicate_groups = [
        policies
        for policies in groups.values()
        if len(policies) >= 2 and policies[0]["policy_name"]
    ]
    duplicate_groups.sort(
        key=lambda policies: (-len(policies), policies[0]["policy_name"])
    )
    cases = duplicate_groups[: args.limit]
    if len(cases) < args.limit:
        raise RuntimeError(
            f"평가 가능한 중복 정책명 그룹이 부족합니다: {len(cases)}/{args.limit}"
        )

    service = PolicySimilarityService(corpus=corpus)
    hits = 0
    reciprocal_ranks = []
    latencies = []
    case_results = []

    for policies in cases:
        query_policy = policies[0]
        relevant_ids = {policy["service_id"] for policy in policies[1:]}
        result = service.search(
            as_pipeline_policy(query_policy),
            top_k=args.top_k,
            min_score=0,
        )
        returned_ids = [item["service_id"] for item in result["results"]]
        first_rank = next(
            (
                rank
                for rank, service_id in enumerate(returned_ids, start=1)
                if service_id in relevant_ids
            ),
            None,
        )
        if first_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / first_rank)
        else:
            reciprocal_ranks.append(0.0)
        latencies.append(result["query_time_ms"])
        case_results.append(
            {
                "policy_name": query_policy["policy_name"],
                "relevant_count": len(relevant_ids),
                "first_relevant_rank": first_rank,
                "returned_ids": returned_ids,
            }
        )

    recall_at_k = hits / len(cases)
    warm_latencies = latencies[1:] or latencies
    report = {
        "case_count": len(cases),
        "top_k": args.top_k,
        "recall_at_k": round(recall_at_k, 4),
        "mrr_at_k": round(statistics.mean(reciprocal_ranks), 4),
        "cold_start_ms": round(latencies[0], 1),
        "warm_latency_mean_ms": round(statistics.mean(warm_latencies), 1),
        "warm_latency_p95_ms": round(percentile_95(warm_latencies), 1),
        "cases": case_results,
    }
    if args.summary_only:
        report.pop("cases")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if recall_at_k < args.min_recall:
        raise SystemExit(
            f"Recall@{args.top_k} {recall_at_k:.3f}가 기준 {args.min_recall:.3f} 미만입니다."
        )


if __name__ == "__main__":
    main()
