import os
import shutil
import chromadb

from embedding import embeddings
from document_builder import documents

# 현재 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# 기존 DB 삭제
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

# ChromaDB 생성
client = chromadb.PersistentClient(path=DB_PATH)

# Collection 생성
collection = client.get_or_create_collection(
    name="law_collection"
)

# 문서 저장
for i in range(len(documents)):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[documents[i]]
    )

print("저장 완료!")
print("총 문서 수 :", collection.count())