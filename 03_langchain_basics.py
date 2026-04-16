import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

print("="*50)
print("Gọi LLM trực tiếp")
print("="*50)

result = llm.invoke("Xin chào! Bạn là ai?")
print(result.content)

print("\n" + "="*50)
print("Prompt template")
print("=" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là giáo viên lập trình, giải thích đơn giản, dễ hiểu."),
    ("user", "Giải thích khái niệm '{topic}' trong {language}, Kèm 1 ví dụ code ngắn."),
])

formatted = prompt.invoke({
    "topic": "list comprehension",
    "language": "Python"
})
print(f"Prompt đã format:\n{formatted.messages[-1].content}\n")

result = llm.invoke(formatted)
print(f"Kết quả:\n{result.content}")

print("\n" + "="*50)
print("Chain LCEL")
print("="*50)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "topic": "decorator",
    "language": "Python"
})
print(result)

print("\n" + "="*50)
print("Multi-step chain")
print("="*50)

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là giáo viên. Tạo 3 câu hỏi trắc nghiệm (A/B/C/D)."),
    ("user", "Tạo quiz về chủ đề: {topic}"),
])

quiz_chain = quiz_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là giáo viên. Đưa ra đáp án có giải thích."),
    ("user", "Đây là bài quiz:\n{quiz}\nHãy cho đáp án và giải thích từng câu."),
])

answer_chain = answer_prompt | llm | StrOutputParser()

quiz = quiz_chain.invoke({"topic": "Python Functions"})
print(f"Quiz:\n{quiz}\n")

answers = answer_chain.invoke({"quiz": quiz})
print(f"Đáp án:\n{answers}")