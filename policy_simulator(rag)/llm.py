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

collection = client.get_collection("policy_collection")

# ==========================
# SBERT 모델
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
# Chroma 검색
# ==========================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

docs = results["documents"][0]

# ==========================
# 정책명 기준 재정렬 (Re-ranking)
# ==========================

def score(doc):
    score = 0

    policy_name = ""

    for line in doc.split("\n"):
        if line.startswith("정책명:"):
            policy_name = line.replace("정책명:", "").strip()
            break

    # 질문 단어가 정책명에 포함되면 점수 부여
    for word in query.split():
        if word in policy_name:
            score += 5

    return score

docs = sorted(
    docs,
    key=score,
    reverse=True
)

# LLM에 전달할 문맥
context = "\n\n".join(docs)

# ==========================
# Prompt
# ==========================

messages = [
    {
        "role": "system",
        "content": (
            "당신은 대한민국 정책 안내 AI입니다.\n"
            "반드시 제공된 정책만 참고하여 답변하세요.\n"
            "정책에 없는 내용은 절대 추론하거나 추가하지 마세요.\n"
            "사용자가 이해하기 쉽게 자연스럽게 답변하세요.\n"
            "관련 정책이 없으면 '관련 정책에서 확인되지 않습니다.'라고 답하세요."
        ),
    },
    {
        "role": "user",
        "content": f"""
다음 정책을 참고하여 질문에 답변하세요.

[정책]

{context}

[질문]

{query}

다음 형식을 지켜 답변하세요.

답변:
(질문에 대한 답을 자연스럽게 작성)

근거:
정책명:
지원대상:
신청기간:
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

for i, doc in enumerate(docs, 1):
    print(f"[{i}]")
    print(doc)
    print("-" * 80)

print("=" * 60)