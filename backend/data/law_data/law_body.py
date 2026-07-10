import os
import time
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()
OC = os.getenv("LAW_OC")
SAVE_DIR = "law_bodies"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv("law_list.csv", dtype=str)
mst_col = "법령일련번호"

done = set(f.replace(".xml", "") for f in os.listdir(SAVE_DIR))

fail_list = []

for mst in tqdm(df[mst_col].dropna().unique()):
    if mst in done:
        continue
    url = f"http://www.law.go.kr/DRF/lawService.do?OC={OC}&target=law&MST={mst}&type=XML"
    try:
        res = requests.get(url, timeout=10)
        with open(os.path.join(SAVE_DIR, f"{mst}.xml"), "w", encoding="utf-8") as f:
            f.write(res.text)
    except Exception as e:
        fail_list.append(mst)
        print(f"실패: {mst} - {e}")
    time.sleep(0.3)

print("전체 본문 수집 완료")
print(f"실패 건수: {len(fail_list)}")
if fail_list:
    with open("fail_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(fail_list))
