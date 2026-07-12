import json
import re
import sqlite3

from backend.ai_simulation_core.llm_inference.llm_gateway.models.run_llm import (
    run_llm,
)
from backend.ai_simulation_core.llm_inference.prompts.citizen import citizen_prompt
from backend.scoring.validator import validate_citizen_response


def parse_citizen_response(raw_output: str) -> dict | None:
    # 로컬 모델이 JSON 코드 블록을 붙인 경우 마크다운 기호를 제거
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[JSON 파싱 실패] {e}\n원본: {raw_output[:200]}")
        return None


def run_citizen_simulation(
    persona: dict, policy: dict, max_retries: int = 3
) -> dict | None:
    prompt = citizen_prompt(persona, policy)

    parsed = None
    errors = []

    for attempt in range(max_retries):
        # main과 같은 로컬 Qwen 모델을 사용하여 시민 응답 생성
        raw_output = run_llm(prompt)
        parsed = parse_citizen_response(raw_output)

        if parsed is None:
            print(f"  [시도 {attempt + 1}/{max_retries}] JSON 파싱 실패, 재시도")
            continue

        errors = validate_citizen_response(parsed, persona, policy)
        if not errors:
            if attempt > 0:
                print(f"  [시도 {attempt + 1}/{max_retries}] 검증 통과")
            break
        print(f"  [시도 {attempt + 1}/{max_retries}] 검증 실패: {errors}")

    if parsed is None:
        return None

    # 스코어링과 결과 확인에 필요한 페르소나 식별자와 검증 결과 추가
    parsed["persona_id"] = persona.get("id")
    parsed["_validation_errors"] = errors
    return parsed


def load_test_personas(db_path: str, n: int = 5) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(f"""
        SELECT uuid, sex, age, occupation, district, province,
               family_type, housing_type, education_level, persona
        FROM persona
        ORDER BY RANDOM()
        LIMIT {n}
    """)
    personas = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for p in personas:
        p["id"] = p.pop("uuid")
    return personas


if __name__ == "__main__":
    test_personas = load_test_personas(
        db_path=r"C:\Users\chhs2\korean-people-persona\database\persona.db", n=25
    )

    test_policy = {
        "상세정보": {
            "서비스명": "청년 월세 지원사업",
            "소관기관명": "국토교통부",
            "지원대상": "만 19세~34세 무주택 청년 1인 가구",
            "선정기준": "청년독립가구 기준 중위소득 60% 이하",
            "지원내용": "월 최대 20만원, 최대 12개월",
            "신청방법": "복지로 온라인 신청 또는 관할 주민센터 방문",
            "신청기한": "2026년 12월 31일까지",
        }
    }

    results = []
    for i, persona in enumerate(test_personas):
        print(f"\n=== 페르소나 {i+1}/{len(test_personas)}: {persona.get('occupation')} ({persona.get('age')}세) ===")
        result = run_citizen_simulation(persona, test_policy)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    with open("backend/data/test_outputs/citizen_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    valid_count = sum(1 for r in results if r and not r.get("_validation_errors"))
    print(f"\n=== 최종: {valid_count}/{len(results)}명 검증 통과 ===")
