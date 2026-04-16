import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
import re

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

chunk_texts = [c.page_content for c in chunks]
print(f"Đã cắt thành {len(chunks)} chunks\n")

print("=" * 60)
print("BM25 Search (keyword)")
print("=" * 60)

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens

tokenized_chunks = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

query = "PVI Care bảo hiểm"
tokenized_query = tokenize(query)
bm25_scores = bm25.get_scores(tokenized_query)

bm25_top_indices = np.argsort(bm25_scores)[::-1][:5]

print(f"\nQuery: '{query}'")
print(f"BM25 Results:")
for rank, idx in enumerate(bm25_top_indices):
    print(f"[{rank+1}] Score: {bm25_scores[idx]:.4f} | {chunk_texts[idx][:80]}...")

print("\n" + "=" * 60)
print("Vector Search (Semantic)")
print("=" * 60)

vectorstore = FAISS.from_texts(chunk_texts, embeddings)
vector_results = vectorstore.similarity_search_with_score(query, k=5)

print(f"Query: '{query}'")
print(f"Vector results:")
for rank, (doc, score) in enumerate(vector_results):
    print(f"[{rank+1}] Score: {score:4f} | {doc.page_content[:80]}...")

print("\n" + "=" * 60)
print("Hybrid Search (RRF Fusion)")
print("=" * 60)

def reciprocal_rank_fusion(rankings_list, k=60):
    rrf_scores = {}

    for rankings in rankings_list:
        for rank, doc_id in enumerate(rankings):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank + 1)

    sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def hybrid_search(query, chunk_texts, bm25_index, vectorstore, k=5):
    tokenized_q = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_q)
    bm25_ranking = np.argsort(bm25_scores)[::-1][:k*2].tolist()

    vector_results = vectorstore.similarity_search_with_score(query, k=k*2)
    vector_ranking = []
    for doc, score in vector_results:
        for idx, text in enumerate(chunk_texts):
            if text == doc.page_content:
                vector_ranking.append(idx)
                break
    rrf_results = reciprocal_rank_fusion([bm25_ranking, vector_ranking], k=60)
    return rrf_results[:k]

test_queries = [
    "PVI Care bảo hiểm hạn mức",
    "Chính sách khi nhân viên nghỉ",
    "Vietcombank lương ngày 10",
]

for query in test_queries:
    print(f"Query: '{query}'")
    results = hybrid_search(query, chunk_texts, bm25, vectorstore, k=3)

    print("Hybrid (RRF) Results:")
    for rank, (doc_idx, score) in enumerate(results):
        print(f"[{rank+1}] RRF Score: {score:.5f}")
        print(f"{chunk_texts[doc_idx][:100]}...")
    print("-" * 50)

print("\n" + "=" * 60)
print("HYBRID RAG — Hỏi đáp")
print("=" * 60)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Trả lời câu hỏi CHỈ dựa trên CONTEXT. Trích dẫn cụ thể."""),
    ("user", "CONTEXT:\n{context}\n\nCÂU HỎI: {question}"),
])

rag_chain = rag_prompt | llm | StrOutputParser()

question = "Bảo hiểm PVI Care có hạn mức bao nhiêu cho nhân viên và người thân?"

hybrid_results = hybrid_search(question, chunk_texts, bm25, vectorstore, k=3)
context = "\n\n---\n\n".join(chunk_texts[idx] for idx, _ in hybrid_results)

answer = rag_chain.invoke({"context": context, "question": question})
print(f"\n{question}")
print(f"\n{answer}")