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

loader = TextLoader("data/company_docs.txt", encoding="utf-8")
documents = loader.load()
raw_text = documents[0].page_content
print(f"Tổng ký tự: {len(raw_text)}")

embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)

print("\n" + "="*50)
print("Recursive chunking (Module 1)")
print("=" * 50)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_documents(documents)

print(f"Số chunks: {len(recursive_chunks)}\n")
for i, chunk in enumerate(recursive_chunks[:5]):
    print(f"Chunk {i} ({len(chunk.page_content)} ký tự):")
    print(f"'{chunk.page_content[:80]}...'\n")

print("=" * 60)
print("Sematic chunking (Nâng cao)")
print("=" * 60)

def semantic_chunking(text, embeddings_model, threshold=0.5):
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        for sent in paragraph.split(". "):
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)

    if len(sentences) < 2:
        return [text]

    print(f"Tổng số câu: {len(sentences)}")

    print("Đang embed các câu...")
    sentence_embeddings = embeddings_model.embed_documents(sentences)

    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        vec_a = np.array(sentence_embeddings[i])
        vec_b = np.array(sentence_embeddings[i + 1])
        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        similarities.append(cos_sim)
    
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
            print(f"Cắt lại câu {i + 1} (similarity: {sim:3f})")
        else:
            current_chunk.append(sentences[i + 1])
    
    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks

semantic_chunks = semantic_chunking(raw_text, embeddings, threshold=0.5)

print(f"\nSố chunks: {len(semantic_chunks)}\n")
for i, chunk in enumerate(semantic_chunks[:5]):
    print(f"Chunk {i} ({len(chunk)} ký tự):")
    print(f"'{chunk[:100]}...'\n")

print("=" * 60)
print("So sánh Retrieval quality")
print("=" * 60)

vs_recursive = FAISS.from_documents(recursive_chunks, embeddings)
vs_semantic = FAISS.from_texts(semantic_chunks, embeddings)

test_queries = [
    "Chính sách thử việc như thế nào?",
    "Budget đào tạo mỗi nhân viên bao nhiêu?",
    "Quy trình phỏng vấn gồm mấy vòng?"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")

    results_r = vs_recursive.similarity_search_with_score(query, k=2)
    print(f"Recursive chunking:")
    for doc, score in results_r:
        print(f"Score: {score:4f} | {doc.page_content[:80]}...")

    results_r = vs_semantic.similarity_search_with_score(query, k=2)
    print(f"Semantic chunking:")
    for doc, score in results_r:
        print(f"Score: {score:4f} | {doc.page_content[:80]}...")

    print("-" * 50)

print("=" * 60)
print("HNSW index parameters")
print("=" * 60)

import faiss

dimension = len(embeddings.embed_query("test"))
print(f"Embedding dimension: {dimension}")

hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
hnsw_index.hnsw.efConstruction = 200
hnsw_index.hnsw.efSearch = 100

print(f"M (connections): 32")
print(f"ef_construction: {hnsw_index.hnsw.efConstruction}")
print(f"ef_search: {hnsw_index.hnsw.efSearch}")
print(f"HNSW index đã tạo thành công!")