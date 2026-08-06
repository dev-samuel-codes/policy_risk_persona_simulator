import torch
from sentence_transformers import SentenceTransformer
from document_builder import documents

# ==========================
# SBERT 모델
# ==========================

model = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

# GPU 사용 가능하면 GPU
if torch.cuda.is_available():
    model = model.to("cuda")

# ==========================
# 임베딩
# ==========================

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ==========================
# 확인
# ==========================

print(f"Document 개수 : {len(documents)}")
print(f"Embedding 개수 : {len(embeddings)}")
print(f"Embedding Shape : {embeddings.shape}")