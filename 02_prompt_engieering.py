import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

def call_llm(messages, temperature=0.7):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1000
    )
    return response.choices[0].message.content

print("=" * 50)
print("Zero-shot prompting")
print("=" * 50)

result = call_llm([
    {"role": "user", 
    "content": "Phân loại sentiment (Positive/Negative/Neutral): 'Sản phẩm này quá tệ, tôi rất thất vọng'"}
])
print(result)

print("\n" + "=" * 50)
print("Few-shot prompting")
print("=" * 50)

result = call_llm([
    {"role": "user",
    "content": """Phân loại sentiment cho các câu sau.

Ví dụ:
- "Tôi rất thích sản phẩm này" -> Positive
- "Giao hàng quá chậm" -> Negative
- "Sản phẩm bình thường" -> Neutral

Bây giờ phân loại:
- "Chất lượng ổn nhưng giá hơi cao"
- "Tuyệt vời! Sẽ mua lại lần nữa"
- "Không bao giờ mua lại, quá tệ"
"""}
])
print(result)

print("/n" + "="*50)
print("Chain-of-thought prompting")
print("="*50)

result = call_llm([
    {"role": "user",
    "content": """Một cửa hàng có 23 quả táo. Họ dùng 8 quả để làm nước ép. 
Sau đó nhập thêm 15 quả. Cuối ngày bán được 12 quả. 
Hỏi cuối ngày còn bao nhiêu quả táo?

Hãy suy nghĩ từng bước (step-by-step) trước khi đưa ra đáp án cuối cùng
"""}
])
print(result)

print("\n" + "="*50)
print("Role prompting + structured output")
print("="*50)

result = call_llm([
    {
        "role": "system",
        "content": """Bạn là chuyên gia phân tích dữ liệu.
Luôn trả lời dưới dạng JSON với format:
{
    "summary": "tóm tắt ngắn",
    "key_points": ["điểm 1", "điểm 2"],
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0
}"""
    },
    {
        "role": "user",
        "content": "Phân tích review sau: 'Khách sạn sạch sẽ, nhân viên thân thiện, nhưng đồ ăn hơi nhạt. Giá cả hợp lý cho vị trí trung tâm.'"
    }
], temperature=0.3)
print(result)