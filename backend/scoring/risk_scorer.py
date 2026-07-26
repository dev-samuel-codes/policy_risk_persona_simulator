from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_CONFIG_PATH = PROJECT_ROOT / "config" / "civil_complaint_risk.yaml"

# 반올림 자릿수 (매직넘버 제거)
INDEX_DECIMALS = 3
SCORE_DECIMALS = 1

# 위험도 컷 기준: (하한 점수, 등급명) 튜플을 점수 내림차순으로 정의.
# 구간을 추가/변경할 땐 이 리스트에 튜플만 추가하면 됨 (if-elif 수정 불필요).
RISK_LEVEL_TABLE: list[tuple[float, str]] = [
    (70, "위험"),
    (40, "보통"),
    (0, "괜찮음"),
]


@dataclass
class RiskScoreResult:
    index: float
    score: float
    risk_level: str
    rate_by_category: dict[str, float]
    n_personas: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_risk_pack(path: Path = RISK_CONFIG_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)["risk_categories"]


def classify_risk_level(score: float) -> str:
    """0~100 점수를 받아 RISK_LEVEL_TABLE 기준으로 등급을 반환한다."""
    for threshold, level in RISK_LEVEL_TABLE:
        if score >= threshold:
            return level
    return RISK_LEVEL_TABLE[-1][1]


def _count_categories(classified_results: list[dict], category_ids: set[str]) -> dict[str, int]:
    """페르소나별 중복 카테고리는 1회만 집계."""
    counts = {c: 0 for c in category_ids}
    for persona_result in classified_results:
        seen = {
            complaint["risk_category"]
            for complaint in persona_result["complaints"]
        }
        for cat in seen & category_ids:
            counts[cat] += 1
    return counts


def compute_index(classified_results: list[dict], risk_pack: list[dict]) -> dict:
    n_personas = len(classified_results)
    weights = {r["id"]: r["weight"] for r in risk_pack}

    category_counts = _count_categories(classified_results, set(weights))
    rate = {c: category_counts[c] / n_personas for c in weights}

    index = sum(weights[c] * rate[c] for c in weights)
    score = round(index * 100, SCORE_DECIMALS)

    result = RiskScoreResult(
        index=round(index, INDEX_DECIMALS),
        score=score,
        risk_level=classify_risk_level(score),
        rate_by_category={c: round(r, INDEX_DECIMALS) for c, r in rate.items()},
        n_personas=n_personas,
    )
    return result.to_dict()
