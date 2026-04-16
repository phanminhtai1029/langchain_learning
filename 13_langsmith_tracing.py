"""
Bài 13: LangSmith Observability
Mục tiêu: Tracing tự động, theo dõi latency, tokens, debug
Thư viện: langsmith, langchain-google-genai
API: Google Gemini
"""

import os
import time
from dotenv import load_dotenv

# ⚡ Dùng Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langsmith import Client as LangSmithClient

load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ============================
# BƯỚC 1: Enable LangSmith Tracing
# Chỉ cần set environment variables → TỰ ĐỘNG trace!
# ============================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "learn-agent-module3"  # Tên project trên LangSmith

print("=" * 60)
print("📌 LANGSMITH TRACING — ENABLED")
print("=" * 60)
print(f"  Project: {os.environ['LANGCHAIN_PROJECT']}")
print(f"  Dashboard: https://smith.langchain.com\n")


# ============================
# BƯỚC 2: Setup RAG Pipeline (Gemini)
# Nhưng giờ MỌI THỨ TỰ ĐỘNG ĐƯỢC TRACE!
# ============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

loader = TextLoader("data/company_docs.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Trả lời câu hỏi CHỈ dựa trên CONTEXT. Ngắn gọn, chính xác."),
    ("user", "CONTEXT:\n{context}\n\nCÂU HỎI: {question}"),
])


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)


# ============================
# BƯỚC 3: Chạy RAG — Tự động trace trên LangSmith!
# ============================
print("=" * 60)
print("📌 BƯỚC 3: Chạy RAG (traces tự động gửi lên LangSmith)")
print("=" * 60)

questions = [
    "Nhân viên chính thức được bao nhiêu ngày phép?",
    "Lương OT ngày lễ bao nhiêu phần trăm?",
    "Career path cho lập trình viên?",
]

for q in questions:
    print(f"\n❓ {q}")
    start = time.time()
    answer = rag_chain.invoke(q)
    elapsed = time.time() - start
    print(f"✅ {answer[:100]}...")
    print(f"⏱️  Latency: {elapsed:.2f}s")


# ============================
# BƯỚC 4: @traceable — Trace custom functions
# Dùng cho hàm KHÔNG phải LangChain
# ============================
print("\n\n" + "=" * 60)
print("📌 BƯỚC 4: @traceable — Custom function tracing")
print("=" * 60)


@traceable(name="custom_rag_pipeline", run_type="chain")
def custom_rag(question: str) -> dict:
    """Custom RAG pipeline — traced tự động nhờ @traceable."""

    # Step 1: Retrieve
    docs = retrieve_documents(question)

    # Step 2: Re-rank (đơn giản)
    ranked_docs = simple_rerank(question, docs)

    # Step 3: Generate
    answer = generate_answer(question, ranked_docs)

    return {
        "question": question,
        "answer": answer,
        "num_contexts": len(ranked_docs),
    }


@traceable(name="retrieve_documents", run_type="retriever")
def retrieve_documents(question: str) -> list:
    """Retrieve relevant documents."""
    docs = retriever.invoke(question)
    return docs


@traceable(name="simple_rerank", run_type="tool")
def simple_rerank(question: str, docs: list) -> list:
    """Simple re-ranking: giữ top 2 docs dài nhất."""
    sorted_docs = sorted(docs, key=lambda d: len(d.page_content), reverse=True)
    return sorted_docs[:2]


@traceable(name="generate_answer", run_type="llm")
def generate_answer(question: str, docs: list) -> str:
    """Generate answer from context."""
    context = format_docs(docs)
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# Chạy custom pipeline — trace sẽ hiện nested trên LangSmith!
result = custom_rag("Fresher được đào tạo những gì?")
print(f"\n❓ {result['question']}")
print(f"✅ {result['answer'][:150]}...")
print(f"📊 Contexts used: {result['num_contexts']}")


# ============================
# BƯỚC 5: Tạo Dataset trên LangSmith (cho future evaluation)
# ============================
print("\n\n" + "=" * 60)
print("📌 BƯỚC 5: Tạo Dataset trên LangSmith")
print("=" * 60)

client = LangSmithClient()

# Tạo dataset
dataset_name = "company-qa-test-v1"

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test dataset for company docs RAG evaluation",
    )
    print(f"✅ Dataset '{dataset_name}' đã tạo trên LangSmith!")

    # Thêm examples
    test_examples = [
        {
            "input": {"question": "Nhân viên chính thức được bao nhiêu ngày phép?"},
            "output": {"answer": "15 ngày phép/năm, tăng 1 ngày mỗi 3 năm thâm niên, tối đa 20 ngày."},
        },
        {
            "input": {"question": "Lương OT ngày lễ tính bao nhiêu phần trăm?"},
            "output": {"answer": "300%."},
        },
        {
            "input": {"question": "Budget đào tạo cho mỗi nhân viên là bao nhiêu?"},
            "output": {"answer": "10 triệu VNĐ/năm."},
        },
        {
            "input": {"question": "Quy trình phỏng vấn gồm mấy vòng?"},
            "output": {"answer": "4 vòng: sàng lọc CV, test kỹ thuật online, phỏng vấn kỹ thuật, phỏng vấn văn hóa."},
        },
    ]

    for ex in test_examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=ex["input"],
            outputs=ex["output"],
        )

    print(f"✅ Đã thêm {len(test_examples)} examples vào dataset!")

except Exception as e:
    print(f"⚠️ Dataset có thể đã tồn tại: {e}")


# ============================
# Đợi traces gửi xong
# ============================
print("\n\n⏳ Đợi traces gửi lên LangSmith...")
try:
    from langsmith import utils as ls_utils
    ls_utils.wait_for_all_tracers()
except Exception:
    time.sleep(5)

print(f"\n🎉 XONG! Vào LangSmith dashboard để xem traces:")
print(f"   🔗 https://smith.langchain.com")
print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}")
print(f"\n💡 Trên dashboard bạn sẽ thấy:")
print(f"   - Trace tree cho mỗi lần gọi")
print(f"   - Latency của từng step")
print(f"   - Token usage & cost estimate")
print(f"   - Input/Output của mỗi LLM call")
