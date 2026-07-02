# 모델 호출 파일

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# mps는 맥북용 GPU가속 장치
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("사용장치: " + device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

#윈도우면 cpu 또는 cuda로 변경
model: Any = AutoModelForCausalLM.from_pretrained(
    model_name,
    type=torch.float16 if device == "mps" else torch.float32,
)

model.to(device)
model.eval()

prompt = "Give me a short introduction to large language model."

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": prompt,
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)