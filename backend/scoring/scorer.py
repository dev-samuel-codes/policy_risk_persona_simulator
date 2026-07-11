import yaml
import json


def load_risk_pack(path="risk-packs/civil-complaint-risk.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["risk_categories"]


def compute_index(classified_results: list[dict], risk_pack: list[dict]) -> dict:
    n_personas = len(classified_results)
    weights = {r["id"]: r["weight"] for r in risk_pack}
    category_counts = {r["id"]: 0 for r in risk_pack}

    for persona_result in classified_results:
        seen_categories = set()
        for complaint in persona_result["complaints"]:
            cat = complaint["risk_category"]
            if cat not in seen_categories:
                category_counts[cat] += 1
                seen_categories.add(cat)

    rate = {c: category_counts[c] / n_personas for c in weights}
    index = sum(weights[c] * rate[c] for c in weights)

    return {
        "index": round(index, 3),
        "rate_by_category": {c: round(r, 3) for c, r in rate.items()},
        "n_personas": n_personas,
    }


if __name__ == "__main__":
    from backend.scoring.classifier import add_risk_categories

    risk_pack = load_risk_pack()
    with open("backend/data/test_outputs/citizen_sample.json", encoding="utf-8") as f:
        results = json.load(f)

    classified_results = add_risk_categories(results)

    with open("backend/data/test_outputs/citizen_sample_classified.json", "w", encoding="utf-8") as f:
        json.dump(classified_results, f, ensure_ascii=False, indent=2)

    result = compute_index(classified_results, risk_pack)
    print(json.dumps(result, ensure_ascii=False, indent=2))
     # ===== 최종 리포트 출력 =====
    weights = {r["id"]: r["weight"] for r in risk_pack}
    names = {r["id"]: r["name"] for r in risk_pack}

    print("\n" + "=" * 50)
    print("정책 민원 리스크 스코어링 결과")
    print("=" * 50)
    print(f"분석 대상 페르소나 수: {result['n_personas']}명")
    print(f"\n【 총 민원 리스크 점수 】  {result['index']:.3f}  (100점 만점 환산: {result['index']*100:.1f}점)")
    print("\n카테고리별 상세:")

    sorted_categories = sorted(
        result["rate_by_category"].items(), key=lambda x: x[1], reverse=True
    )
    for category_id, rate in sorted_categories:
        weight = weights.get(category_id, 0)
        contribution = weight * rate
        name = names.get(category_id, category_id)
        bar = "█" * int(rate * 20)
        print(f"  {name:12s} | rate={rate:.2f} | 가중치={weight:.2f} | 기여도={contribution:.3f} | {bar}")

    print("=" * 50)