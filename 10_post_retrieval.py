import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)
llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)
loader = TextLoader("data/company_docs.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)

print("=" * 60)
print("KỸ THUẬT 1: LLM RE-RANKING")
print("=" * 60)

rerank_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là hệ thống đánh giá mức độ liên quan.
Cho một câu hỏi và một đoạn văn bản, hãy đánh giá mức độ liên quan từ 0 đến 10.
Chỉ trả về MỘT SỐ duy nhất (0-10), không giải thích.

Tiêu chí:
- 0: Hoàn toàn không liên quan
- 5: Có liên quan một phần
- 10: Trả lời trực tiếp câu hỏi"""),
    ("user", "Câu hỏi: {question}\n\nĐoạn văn bản:\n{document}"),
])
rerank_chain = rerank_prompt | llm | StrOutputParser()

def llm_rerank(question, documents, top_k=3):
    scored_docs = []
    for doc in documents:
        try:
            score_text = rerank_chain.invoke({
                "question": question,
                "document": doc.page_content,
            })
            score = float(score_text.strip().split()[0])
            score = min(max(score, 0), 10)
        except (ValueError, IndexError):
            score=0
        
        scored_docs.append((doc, score))
        print(f"Score: {score:.0f}/10 | {doc.page_content[:60]}...")

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]

question = "Nhân viên có được hỗ trợ tiền ăn khi làm OT không?"

print(f"\n'{question}'\n")
print("Vector Search (top 6 - raw results):")
raw_results = vectorstore.similarity_search(question, k=6)
for i, doc in enumerate(raw_results):
    print(f"[{i}] {doc.page_content[:80]}...")

print(f"\nLLM Re-ranking (chọn top 3:)")
reranked = llm_rerank(question, raw_results, top_k=3)

print(f"\nSau re-ranking (top 3):")
for i, (doc, score) in enumerate(reranked):
    print(f"[{i}] Score: {score:.0f}/10 | {doc.page_content[:80]}...")

print("\n\n" + "=" * 60)
print("KỸ THUẬT 2: MMR — Đa dạng hóa kết quả")
print("=" * 60)

question = "Chế độ phúc lợi cho nhân viên?"

print(f"\n '{question}'\n")

print("Similarity Search (có thể trùng lặp):")
sim_results = vectorstore.similarity_search(question, k=4)
for i, doc in enumerate(sim_results):
    print(f"[{i}] {doc.page_content[:80]}...")

print(f"\nMMR Search (đa dạng hơn):")
mmr_results = vectorstore.max_marginal_relevance_search(
    question,
    k=4,
    fetch_k=10,
    lambda_mult=0.5,
)
for i, doc in enumerate(mmr_results):
    print(f"[{i}] {doc.page_content[:80]}...")

print(f"\nSo sánh lambda values:")
for lam in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = vectorstore.max_marginal_relevance_search(
        question, k=3, fetch_k=10, lambda_mult=lam,
    )
    print(f"\n lambda = {lam}")
    for doc in results:
        print(f"--> {doc.page_content[:60]}...")

print("\n\n" + "=" * 60)
print("FULL PIPELINE: Retrieve → Re-rank → Generate")
print("=" * 60)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Trả lời câu hỏi CHỈ dựa trên CONTEXT. Trả lời có cấu trúc, chi tiết."),
    ("user", "CONTEXT:\n{context}\n\nCÂU HỎI: {question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()

def full_rag_pipeline(question, vectorstore, llm, fetch_k=8, rerank_k=3):
    print(f"\n'{question}'")

    print(f"\nStep 1: Retrieve top {fetch_k}...")
    raw_docs = vectorstore.similarity_search(question, k=fetch_k)

    print(f"\nStep 2: Re-rank --> top {rerank_k}...")
    reranked = llm_rerank(question, raw_docs, top_k=rerank_k)

    print(f"\nStep 3: Generate answer...")
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in reranked)
    answer = rag_chain.invoke({"context": context, "question": question})

    return answer

answer = full_rag_pipeline(
    "Quy trình onboarding cho nhân viên mới gồm những gì?",
    vectorstore, llm
)

print(f"\nCâu trả lời:\n{answer}")