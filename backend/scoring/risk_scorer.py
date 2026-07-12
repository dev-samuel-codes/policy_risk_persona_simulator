from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_CONFIG_PATH = PROJECT_ROOT / "config" / "civil_complaint_risk.yaml"


def load_risk_pack(path: Path = RISK_CONFIG_PATH):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)["risk_categories"]


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
