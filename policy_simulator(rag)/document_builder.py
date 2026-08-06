import json
import os

# ==========================
# JSON 읽기
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "policy_conditions.json")

with open(json_path, "r", encoding="utf-8") as f:
    policies = json.load(f)

documents = []

# ==========================
# Document 생성
# ==========================

for policy in policies:

    fields = []

    keys = [
        "정책명",
        "지원대상",
        "신청기간",
        "시행일",
        "제출서류",
        "신청방법",
        "문의처",
        "혜택",
        "제외조건"
    ]

    for key in keys:

        value = policy.get(key)

        # None이면 빈 문자열
        if value is None:
            value = ""

        value = str(value).strip()

        # 값이 있을 때만 추가
        if value:
            fields.append(f"{key}: {value}")

    documents.append("\n".join(fields))

# ==========================
# 확인
# ==========================

print(f"Document 개수 : {len(documents)}")

if documents:
    print("\n첫 번째 Document\n")
    print(documents[0])