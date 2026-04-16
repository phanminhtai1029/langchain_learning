import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()
raw_text = documents[0].page_content

embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"}
)

print("="*50)
print("So sánh chunking strategies")
print("="*50)

splitter_fixed = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=20,
)
chunks_fixed = splitter_fixed.split_documents(documents)

splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks_recursive = splitter_recursive.split_documents(documents)

print(f"\nFixed-size: {len(chunks_fixed)} chunks")
for i, c in enumerate(chunks_fixed[:3]):
    print(f"[{i}] {c.page_content[:80]}...")

print(f"\nRecursive: {len(chunks_recursive)} chunks")
for i, c in enumerate(chunks_recursive[:3]):
    print(f"[{i}] {c.page_content[:80]}...")

print("\n" + "="*50)
print("So sánh retrieval methods")
print("\n" + "="*50)

vectorstore = FAISS.from_documents(chunks_recursive, embeddings)
query = "Chính sách lương thưởng như thế nào?"

print(f"\nQuery: '{query}")
print(f"\nSimilarity Search (top 3):")
result_sim = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(result_sim):
    print(f"[{i}] {doc.page_content[:100]}...")

print(f"\nSimilarity Search with scores:")
results_scored = vectorstore.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(results_scored):
    print(f"[{i}] Score: {score:4f} | {doc.page_content[:80]}...")

print(f"\nMMR Search (diverse results):")
results_mmr = vectorstore.max_marginal_relevance_search(
    query, k=3, fetch_k=6
)
for i, doc in enumerate(results_mmr):
    print(f"[{i}] {doc.page_content[:100]}...")