"""
Bài 14: LangFuse Observability
Mục tiêu: Tracing với LangFuse — CallbackHandler approach
Thư viện: langfuse, langchain-google-genai
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

# LangFuse imports
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ============================
# BƯỚC 1: Tắt LangSmith (tránh trùng), bật LangFuse
# ============================
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Tắt LangSmith

# LangFuse v4: auth qua env vars (đã có trong .env, load_dotenv() đã load)
# Nếu .env dùng LANGFUSE_BASE_URL thì set lại thành LANGFUSE_HOST
if os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")

print("=" * 60)
print("📌 LANGFUSE TRACING — Setup")
print("=" * 60)

# LangFuse v4: CallbackHandler tự đọc env vars
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
langfuse_handler = LangfuseCallbackHandler()

print("✅ LangFuse handler ready!")
print(f"   Dashboard: https://cloud.langfuse.com\n")


# ============================
# BƯỚC 2: Setup RAG Pipeline (Gemini)
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
# BƯỚC 3: Chạy RAG VỚI LANGFUSE TRACING
# Khác LangSmith: phải truyền callback handler vào config!
# ============================
print("=" * 60)
print("📌 BƯỚC 3: Chạy RAG với LangFuse tracing")
print("=" * 60)

questions = [
    "Nhân viên chính thức được bao nhiêu ngày phép?",
    "Chính sách WFH như thế nào?",
    "Fresher được trả lương bao nhiêu?",
    "Giờ làm việc linh hoạt ra sao?",
]

for q in questions:
    print(f"\n❓ {q}")
    start = time.time()

    # Truyền callback handler qua config
    answer = rag_chain.invoke(
        q,
        config={"callbacks": [langfuse_handler]},  # ← Khác LangSmith!
    )

    elapsed = time.time() - start
    print(f"✅ {answer[:100]}...")
    print(f"⏱️  Latency: {elapsed:.2f}s")


# ============================
# BƯỚC 4: Custom Trace với LangFuse SDK v4
# V4 dùng start_as_current_observation() context manager
# ============================
print("\n\n" + "=" * 60)
print("📌 BƯỚC 4: Custom Trace với LangFuse SDK (v4)")
print("=" * 60)

from langfuse import Langfuse

# LangFuse v4: tự đọc env vars LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
langfuse = Langfuse()

# Tạo custom trace bằng context manager (v4 API)
# start_as_current_observation → tự tạo trace + span
with langfuse.start_as_current_observation(
    name="custom-rag-pipeline",
    as_type="chain",
    input={"query": "Budget đào tạo cho nhân viên?"},
    metadata={"module": "3", "experiment": "langfuse-demo"},
) as root_span:

    # Span 1: Retrieval
    with langfuse.start_as_current_observation(
        name="retrieve",
        as_type="retriever",
        input={"query": "Budget đào tạo?"},
    ) as retrieve_span:
        docs = retriever.invoke("Budget đào tạo?")
        retrieve_span.update(
            output={"num_docs": len(docs), "contents": [d.page_content[:50] for d in docs]},
        )

    # Span 2: Generation (ghi nhận LLM call)
    with langfuse.start_as_current_observation(
        name="llm-call",
        as_type="generation",
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Budget đào tạo cho nhân viên?"}],
    ) as gen_span:
        context = format_docs(docs)
        chain = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "question": "Budget đào tạo cho nhân viên?"},
        )
        gen_span.update(output=answer)

    root_span.update(output={"answer": answer})

    # Score trace (đánh giá chất lượng) — phải nằm TRONG context manager
    langfuse.score_current_trace(
        name="relevance",
        value=0.9,
        comment="Answer correctly mentions 10 triệu VNĐ/năm",
    )

print(f"✅ Custom trace created!")
print(f"   Answer: {answer[:100]}...")


# ============================
# BƯỚC 5: Flush — Đảm bảo gửi hết traces
# ============================
langfuse.flush()

print(f"\n🎉 XONG! Vào LangFuse dashboard để xem traces:")
print(f"   🔗 https://cloud.langfuse.com")
print(f"\n💡 Trên dashboard bạn sẽ thấy:")
print(f"   - Traces tab: danh sách tất cả traces")
print(f"   - Click vào trace → thấy tree view nested")
print(f"   - Sessions tab: traces nhóm theo session")
print(f"   - Generations tab: tất cả LLM calls")
print(f"   - Scores tab: đánh giá chất lượng")

