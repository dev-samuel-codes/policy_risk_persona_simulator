from sentence_transformers import SentenceTransformer
from document_builder import documents

# 임베딩 모델 불러오기
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 문서 임베딩
embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Document 개수 : {len(documents)}")
print(f"Embedding 개수 : {len(embeddings)}")
print(f"Embedding Shape : {embeddings.shape}")
