import ollama
import json
import re
import sqlite3
from backend.ai_simulation_core.llm_inference.prompts.citizen_v2 import citizen_prompt
from backend.ai_simulation_core.llm_inference.validator import validate_citizen_response

def parse_citizen_response(raw_output: str) -> dict:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[JSON 파싱 실패] {e}\n원본: {raw_output[:200]}")
        return None


def _call_ollama(prompt: str, model: str, temperature: float) -> dict:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return parse_citizen_response(response["message"]["content"])


def run_citizen_simulation(
    persona: dict, policy: dict, model: str = "qwen2.5:7b", max_retries: int = 3
) -> dict:
    prompt = citizen_prompt(persona, policy)
    temperatures = [0.6, 0.4, 0.2]  # 재시도할수록 더 보수적으로

    parsed = None
    for attempt in range(max_retries):
        temp = temperatures[min(attempt, len(temperatures) - 1)]
        parsed = _call_ollama(prompt, model, temp)

        if parsed is None:
            print(f"  [시도 {attempt + 1}/{max_retries}] JSON 파싱 실패, 재시도")
            continue

        errors = validate_citizen_response(parsed, persona, policy)
        if not errors:
            if attempt > 0:
                print(f"  [시도 {attempt + 1}/{max_retries}] 검증 통과")
            break
        else:
            print(f"  [시도 {attempt + 1}/{max_retries}] 검증 실패: {errors}")

    if parsed:
        parsed["persona_id"] = persona.get("id")
        parsed["_validation_errors"] = errors if 'errors' in dir() else []

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
        db_path=r"C:\Users\chhs2\korean-people-persona\database\persona.db", n=5
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