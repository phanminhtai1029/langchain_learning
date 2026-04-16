"""
Bài 15: Experiment Comparison — Naive vs Hybrid vs HyDE
Mục tiêu: Chạy 3 RAG architectures, đo bằng Ragas, so sánh
API: Google Gemini
"""

import os
import re
import time
import numpy as np
from dotenv import load_dotenv

# ⚡ Dùng Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

# Ragas
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangSmith tracing
from langsmith import traceable

load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Enable LangSmith cho experiment tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "rag-experiment-comparison"


# ============================
# SETUP CHUNG (Gemini)
# ============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# Load & index
loader = TextLoader("data/company_docs.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
chunk_texts = [c.page_content for c in chunks]

vectorstore = FAISS.from_documents(chunks, embeddings)

# BM25 cho Hybrid
tokenized_chunks = [re.findall(r'\w+', t.lower()) for t in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

# RAG Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Trả lời câu hỏi CHỈ dựa trên CONTEXT. Ngắn gọn, chính xác."),
    ("user", "CONTEXT:\n{context}\n\nCÂU HỎI: {question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()

# HyDE prompt
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "Viết đoạn trả lời chi tiết (100 từ) như trích từ sổ tay nhân viên công ty."),
    ("user", "{question}"),
])
hyde_chain = hyde_prompt | llm | StrOutputParser()


# ============================
# TEST DATASET
# ============================
test_cases = [
    {
        "question": "Nhân viên chính thức được bao nhiêu ngày phép một năm?",
        "ground_truth": "15 ngày phép/năm, tăng 1 ngày mỗi 3 năm thâm niên, tối đa 20 ngày.",
    },
    {
        "question": "Lương OT ngày lễ tính bao nhiêu phần trăm?",
        "ground_truth": "Lương OT ngày lễ tính 300%.",
    },
    {
        "question": "Budget đào tạo cho mỗi nhân viên là bao nhiêu?",
        "ground_truth": "10 triệu VNĐ/năm.",
    },
    {
        "question": "Bảo hiểm PVI Care hạn mức cho nhân viên?",
        "ground_truth": "30 triệu/năm cho nhân viên, 15 triệu/năm cho người thân.",
    },
    {
        "question": "Chính sách WFH ra sao?",
        "ground_truth": "Tối đa 2 ngày/tuần, cần đăng ký trên hệ thống HRMS trước 1 ngày.",
    },
]


# ============================
# 3 RAG ARCHITECTURES
# ============================

@traceable(name="naive_rag", run_type="chain")
def naive_rag(question: str) -> tuple:
    """Architecture 1: Naive RAG — vector search đơn giản."""
    docs = vectorstore.similarity_search(question, k=3)
    contexts = [d.page_content for d in docs]
    context_text = "\n\n---\n\n".join(contexts)
    response = rag_chain.invoke({"context": context_text, "question": question})
    return response, contexts


@traceable(name="hybrid_rag", run_type="chain")
def hybrid_rag(question: str) -> tuple:
    """Architecture 2: Hybrid RAG — BM25 + Vector + RRF."""
    # BM25 search
    tokenized_q = re.findall(r'\w+', question.lower())
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_top = np.argsort(bm25_scores)[::-1][:6].tolist()

    # Vector search
    vector_results = vectorstore.similarity_search_with_score(question, k=6)
    vector_top = []
    for doc, _ in vector_results:
        for idx, text in enumerate(chunk_texts):
            if text == doc.page_content:
                vector_top.append(idx)
                break

    # RRF fusion
    rrf_scores = {}
    for rankings in [bm25_top, vector_top]:
        for rank, doc_id in enumerate(rankings):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (60 + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sorted_results[:3]]

    contexts = [chunk_texts[i] for i in top_indices]
    context_text = "\n\n---\n\n".join(contexts)
    response = rag_chain.invoke({"context": context_text, "question": question})
    return response, contexts


@traceable(name="hyde_rag", run_type="chain")
def hyde_rag(question: str) -> tuple:
    """Architecture 3: HyDE RAG — hypothetical document embedding."""
    # Generate hypothetical document
    hypothetical = hyde_chain.invoke({"question": question})

    # Clean text to prevent 500 Internal error from Google API embedder
    clean_hypothetical = hypothetical.replace('\n', ' ').strip()[:500]

    # Search using hypothetical document (not the question!)
    docs = vectorstore.similarity_search(clean_hypothetical, k=3)
    contexts = [d.page_content for d in docs]
    context_text = "\n\n---\n\n".join(contexts)
    response = rag_chain.invoke({"context": context_text, "question": question})
    return response, contexts


# ============================
# CHẠY THÍ NGHIỆM
# ============================
print("=" * 60)
print("🧪 EXPERIMENT COMPARISON: Naive vs Hybrid vs HyDE")
print("=" * 60)

architectures = {
    "Naive RAG": naive_rag,
    "Hybrid RAG": hybrid_rag,
    "HyDE RAG": hyde_rag,
}

all_results = {}

for arch_name, arch_fn in architectures.items():
    print(f"\n\n{'='*60}")
    print(f"🔬 Running: {arch_name}")
    print(f"{'='*60}")

    samples = []
    total_time = 0

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        ground_truth = tc["ground_truth"]

        start = time.time()
        
        # Thêm cơ chế tự động thử lại với Exponential Backoff (10s, 20s, 40s...)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = 10 * attempt
                    print(f"      ⏳ Đang tạm nghỉ {delay}s cho API hồi phục...")
                    time.sleep(delay) 
                response, contexts = arch_fn(question)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"      ⚠️ API Lỗi (thử lại {attempt+1}/{max_retries}): {e}")
        
        elapsed = time.time() - start
        total_time += elapsed

        print(f"  [{i+1}] Q: {question}")
        print(f"      A: {response[:80]}...")
        print(f"      ⏱️  {elapsed:.2f}s")

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=response,
            reference=ground_truth,
        )
        samples.append(sample)

    # Ragas evaluation
    print(f"\n  ⏳ Evaluating with Ragas...")
    dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_emb = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        LLMContextRecall(llm=evaluator_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)

    all_results[arch_name] = {
        "scores": result,
        "avg_latency": total_time / len(test_cases),
        "df": result.to_pandas(),
    }

    print(f"\n  📊 {arch_name} Results:")
    print(f"  {result}")
    print(f"  Avg Latency: {total_time / len(test_cases):.2f}s")
    
    print("\n  ⏳ Nghỉ 60s trước khi sang experiment mới để API Limit Minute Reset...")
    time.sleep(60)


# ============================
# SO SÁNH KẾT QUẢ
# ============================
print("\n\n" + "=" * 60)
print("📊 BẢNG SO SÁNH TỔNG HỢP")
print("=" * 60)

# Header
print(f"\n{'Metric':<25}", end="")
for arch_name in architectures:
    print(f"{arch_name:<18}", end="")
print()
print("-" * 79)

# Metrics
metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
for metric in metric_names:
    print(f"{metric:<25}", end="")
    for arch_name in architectures:
        df = all_results[arch_name]["df"]
        matching_cols = [c for c in df.columns if metric in c.lower()]
        if matching_cols:
            val = df[matching_cols[0]].mean()
            print(f"{val:<18.3f}", end="")
        else:
            print(f"{'N/A':<18}", end="")
    print()

# Latency
print(f"{'avg_latency (s)':<25}", end="")
for arch_name in architectures:
    val = all_results[arch_name]["avg_latency"]
    print(f"{val:<18.2f}", end="")
print()

# Best architecture
print("\n" + "-" * 79)
print("\n🏆 PHÂN TÍCH:")

for metric in metric_names:
    best_arch = None
    best_score = -1
    for arch_name in architectures:
        df = all_results[arch_name]["df"]
        matching_cols = [c for c in df.columns if metric in c.lower()]
        if matching_cols:
            val = df[matching_cols[0]].mean()
            if val > best_score:
                best_score = val
                best_arch = arch_name
    if best_arch:
        print(f"  {metric}: 🥇 {best_arch} ({best_score:.3f})")

# Overall winner
print("\n📌 TỔNG KẾT:")
overall_scores = {}
for arch_name in architectures:
    df = all_results[arch_name]["df"]
    scores = []
    for metric in metric_names:
        matching_cols = [c for c in df.columns if metric in c.lower()]
        if matching_cols:
            scores.append(df[matching_cols[0]].mean())
    if scores:
        overall_scores[arch_name] = np.mean(scores)

if overall_scores:
    winner = max(overall_scores, key=overall_scores.get)
    print(f"  🏆 Winner: {winner} (avg score: {overall_scores[winner]:.3f})")
    for arch, sc in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"     {'🥇' if arch == winner else '  '} {arch}: {sc:.3f}")

print(f"\n💡 Vào LangSmith project 'rag-experiment-comparison' để xem traces chi tiết!")
