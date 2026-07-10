import time
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()
OC = os.getenv("LAW_OC")

def get_law_list(target="law", display=100):
    rows = []
    page = 1
    while True:
        url = (
            f"http://www.law.go.kr/DRF/lawSearch.do?OC={OC}"
            f"&target={target}&type=XML&display={display}&page={page}"
        )
        res = requests.get(url)
        print("=== 응답 내용 (앞 500자) ===")
        print(res.text[:500])
        print("=========================")
        xtree = ET.fromstring(res.text)

        total_cnt = int(xtree.find("totalCnt").text)
        if page == 1:
            print(f"전체 {total_cnt}건")

        items = xtree.findall("law")
        if not items:
            break

        for node in items:
            row = {child.tag: child.text for child in node}
            rows.append(row)

        page += 1
        time.sleep(0.3)
        if len(rows) >= total_cnt:
            break

    return pd.DataFrame(rows)
df = get_law_list(target="law")
df.to_csv("law_list.csv", index=False, encoding="utf-8-sig")
print(f"저장 완료: {len(df)}건")
