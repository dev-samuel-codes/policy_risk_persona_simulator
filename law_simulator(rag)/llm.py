import os
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# ChromaDB
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("law_collection")

# ==========================
# 임베딩 모델
# ==========================

embed_model = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

# ==========================
# Qwen 모델
# ==========================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# ==========================
# 질문 입력
# ==========================

query = input("질문 : ")

# ==========================
# 질문 임베딩
# ==========================

query_embedding = embed_model.encode(query).tolist()

# ==========================
# ChromaDB 검색
# ==========================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

# 검색된 문서 합치기
context = "\n\n".join(results["documents"][0])

# ==========================
# Prompt
# ==========================

messages = [
    {
        "role": "system",
        "content": (
            "당신은 대한민국 법령 질의응답 AI입니다.\n"
            "반드시 제공된 법령만 참고하여 답변하세요.\n"
            "법령에 없는 내용은 절대 추론하거나 추가하지 마세요.\n"
            "법령 내용을 그대로 복사하지 말고 핵심만 자연스럽게 요약하세요.\n"
            "답을 찾을 수 없는 경우에만 '관련 법령에서 확인되지 않습니다.'라고 답변하세요."
        ),
    },
    {
        "role": "user",
        "content": f"""
다음 법령을 참고하여 질문에 답변하세요.

[법령]

{context}

[질문]

{query}

다음 형식을 반드시 지켜 답변하세요.

답변:
(질문에 대한 답을 한두 문장으로 자연스럽게 작성)

근거:
제○조(조문제목)
"""
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(llm.device)

outputs = llm.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

# ==========================
# 출력
# ==========================

print("\n" + "=" * 60)
print("질문")
print("-" * 60)
print(query)

print("\n답변")
print("-" * 60)
print(response)

print("\n검색 결과(RAG)")
print("-" * 60)

for i, doc in enumerate(results["documents"][0], 1):
    print(f"[{i}]")
    print(doc)
    print("-" * 80)

print("=" * 60)