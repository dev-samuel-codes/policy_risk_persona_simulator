import json

policy_name = input("정책명: ")
target = input("지원대상: ")
period = input("신청기간: ")
start_date = input("시행일: ")
documents = input("제출서류: ")
method = input("신청방법: ")
contact = input("문의처: ")
benefit = input("혜택: ")
excluded = input("제외조건: ")

result = {
    "정책명": policy_name,
    "지원대상": target,
    "신청기간": period,
    "시행일": start_date,
    "제출서류": documents,
    "신청방법": method,
    "문의처": contact,
    "혜택": benefit,
    "제외조건": excluded
}

print("\n===== JSON 결과 =====\n")

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=4
    )
)
