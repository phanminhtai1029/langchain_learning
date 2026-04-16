"""
Bài 12: Ragas Evaluation Metrics
Mục tiêu: Đo chất lượng RAG pipeline bằng thư viện Ragas
Thư viện: ragas (v0.4+), langchain-google-genai
API: Google Gemini
"""

import os
from dotenv import load_dotenv

# ⚡ Dùng Gemini thay OpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ragas imports
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


# ============================
# BƯỚC 1: Setup RAG Pipeline (dùng Gemini)
# ============================
print("=" * 60)
print("📌 BƯỚC 1: Setup RAG Pipeline (Google Gemini)")
print("=" * 60)

# ⚡ LLM: Gemini 2.0 Flash (miễn phí, nhanh)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.3,
)

# ⚡ Embeddings: text-embedding-004 (miễn phí)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# Load & index
loader = TextLoader("data/company_docs.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Trả lời câu hỏi CHỈ dựa trên CONTEXT. Ngắn gọn, chính xác."),
    ("user", "CONTEXT:\n{context}\n\nCÂU HỎI: {question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()


def run_rag(question):
    """Chạy RAG pipeline, trả về (response, contexts)."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n---\n\n".join(contexts)
    response = rag_chain.invoke({"context": context_text, "question": question})
    return response, contexts


print("✅ RAG pipeline ready! (Gemini 2.0 Flash)\n")


# ============================
# BƯỚC 2: Tạo Test Dataset
# ============================
print("=" * 60)
print("📌 BƯỚC 2: Tạo Test Dataset (QA pairs + Ground Truth)")
print("=" * 60)

# Định nghĩa câu hỏi + ground truth (đáp án chuẩn)
test_cases = [
    {
        "question": "Nhân viên chính thức được bao nhiêu ngày phép một năm?",
        "ground_truth": "Nhân viên chính thức được 15 ngày phép/năm, tăng 1 ngày mỗi 3 năm thâm niên, tối đa 20 ngày.",
    },
    {
        "question": "Lương OT ngày lễ tính bao nhiêu phần trăm?",
        "ground_truth": "Lương OT ngày lễ tính 300%.",
    },
    {
        "question": "Budget đào tạo cho mỗi nhân viên là bao nhiêu?",
        "ground_truth": "Mỗi nhân viên có budget đào tạo 10 triệu VNĐ/năm.",
    },
    {
        "question": "Quy trình phỏng vấn gồm mấy vòng?",
        "ground_truth": "Quy trình gồm 4 vòng: sàng lọc CV, bài test kỹ thuật online 90 phút, phỏng vấn kỹ thuật với team lead 60 phút, phỏng vấn văn hóa với HR 30 phút.",
    },
    {
        "question": "Bảo hiểm PVI Care có hạn mức bao nhiêu?",
        "ground_truth": "Bảo hiểm sức khỏe PVI Care với hạn mức 30 triệu/năm cho nhân viên, 15 triệu/năm cho người thân.",
    },
]

# Chạy RAG cho từng câu hỏi
print("\n🔄 Đang chạy RAG cho từng test case...\n")

samples = []
for i, tc in enumerate(test_cases):
    question = tc["question"]
    ground_truth = tc["ground_truth"]

    response, contexts = run_rag(question)

    print(f"  [{i+1}] Q: {question}")
    print(f"      A: {response[:80]}...")
    print(f"      Contexts: {len(contexts)} chunks\n")

    # Tạo SingleTurnSample cho Ragas
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=response,
        reference=ground_truth,
    )
    samples.append(sample)

# Tạo EvaluationDataset
dataset = EvaluationDataset(samples=samples)
print(f"✅ Dataset tạo xong: {len(samples)} samples\n")


# ============================
# BƯỚC 3: Chạy Ragas Evaluation
# ============================
print("=" * 60)
print("📌 BƯỚC 3: Ragas Evaluation")
print("=" * 60)

# Wrap LLM và Embeddings cho Ragas
evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Định nghĩa metrics (RAG Triad + Context Recall)
metrics = [
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
]

print("\n⏳ Đang evaluate (có thể mất vài phút)...\n")

# Chạy evaluation
result = evaluate(
    dataset=dataset,
    metrics=metrics,
)

# ============================
# BƯỚC 4: Phân tích kết quả
# ============================
print("\n" + "=" * 60)
print("📊 KẾT QUẢ EVALUATION")
print("=" * 60)

# In tổng quan
print(f"\n📋 Tổng quan (trung bình):")
print(result)

# Chuyển thành DataFrame để xem chi tiết
df = result.to_pandas()
print(f"\n📋 Chi tiết từng sample:")
print(df.to_string())

# ============================
# BƯỚC 5: Phân tích Component-Wise
# ============================
print("\n\n" + "=" * 60)
print("🔍 PHÂN TÍCH COMPONENT-WISE")
print("=" * 60)

# Tính trung bình riêng cho Retriever vs Generator
# Lọc chỉ các cột metric số (loại bỏ 'retrieved_contexts' chứa list text)
non_metric_cols = {"user_input", "response", "retrieved_contexts", "reference"}
retriever_cols = [c for c in df.columns if "context" in c.lower() and c not in non_metric_cols]
generator_cols = [c for c in df.columns if "faithful" in c.lower() or "relevancy" in c.lower()]

if retriever_cols:
    retriever_avg = df[retriever_cols].mean().mean()
    print(f"\n📌 RETRIEVER Performance: {retriever_avg:.3f}")
    for col in retriever_cols:
        print(f"   {col}: {df[col].mean():.3f}")

if generator_cols:
    generator_avg = df[generator_cols].mean().mean()
    print(f"\n📌 GENERATOR Performance: {generator_avg:.3f}")
    for col in generator_cols:
        print(f"   {col}: {df[col].mean():.3f}")

# Tìm bottleneck
if retriever_cols and generator_cols:
    print("\n💡 PHÂN TÍCH:")
    if retriever_avg < generator_avg:
        print("   ⚠️  RETRIEVER là bottleneck! Cần cải thiện search/indexing.")
        print("   → Thử: Hybrid Search, Semantic Chunking, Query Transformation")
    elif generator_avg < retriever_avg:
        print("   ⚠️  GENERATOR là bottleneck! Cần cải thiện prompt/model.")
        print("   → Thử: Better prompts, Few-shot examples, Stronger model")
    else:
        print("   ✅ Cả 2 tương đương. Pipeline cân bằng!")

# Tìm câu hỏi yếu nhất
print("\n📌 Câu hỏi cần cải thiện (score thấp nhất):")
for col in df.columns:
    if col in ["user_input", "response", "retrieved_contexts", "reference"]:
        continue
    try:
        worst_idx = df[col].idxmin()
        worst_score = df[col].min()
        worst_q = df.loc[worst_idx, "user_input"]
        if worst_score < 0.7:
            print(f"   ⚠️  {col}: {worst_score:.3f} — '{worst_q}'")
    except Exception:
        pass
