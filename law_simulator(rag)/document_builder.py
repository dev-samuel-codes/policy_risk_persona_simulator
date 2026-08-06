import json
import os
import re

# 현재 document_builder.py가 있는 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# JSON 파일 경로
json_path = os.path.join(BASE_DIR, "law_conditions.json")

# JSON 읽기
with open(json_path, "r", encoding="utf-8") as f:
    laws = json.load(f)

documents = []

for law in laws:

    # 조문내용
    content = law.get("조문내용", "")

    # ==========================
    # 전처리
    # ==========================

    # 1. img 태그 제거
    content = re.sub(
        r"<img.*?</img>",
        "",
        content,
        flags=re.DOTALL
    )

    # 2. HTML 태그 제거
    content = re.sub(
        r"<[^>]+>",
        "",
        content
    )

    # 3. 표(Box Drawing 문자)가 있는 줄 제거
    content = re.sub(
        r"^.*[┌┐└┘├┤┬┴┼│─].*$",
        "",
        content,
        flags=re.MULTILINE
    )

    # 4. 공백 정리
    content = re.sub(
        r"\s+",
        " ",
        content
    ).strip()

    # ==========================
    # Document 생성
    # ==========================

    doc = f"""
법령명 : {law.get("법령명", "")}
법령ID : {law.get("법령ID", "")}
공포일자 : {law.get("공포일자", "")}
시행일자 : {law.get("시행일자", "")}
제개정구분 : {law.get("제개정구분", "")}
소관부처 : {law.get("소관부처", "")}
조문번호 : {law.get("조문번호", "")}
조문제목 : {law.get("조문제목", "")}
조문내용 : {content}
"""

    documents.append(doc.strip())
