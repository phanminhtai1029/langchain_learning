import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = api_key
)

print("Đang gọi API tới LLM ...")

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    messages=[
        {
            "role": "system",
            "content" : "Bạn là trợ lý AI chuyên nghiệp, trả lời bằng tiếng Việt."
        },
        {
            "role": "user",
            "content": "AI là gì? Giải thích ngắn gọn trong 3 câu."
        }
    ],
    temperature=0.7,
    max_tokens=500
)

answer = response.choices[0].message.content
print(f"\nAI trả lời:\n{answer}")

print(f"\nModel đã dùng: {response.model}")
print(f"Token dùng: {response.usage.total_tokens}")