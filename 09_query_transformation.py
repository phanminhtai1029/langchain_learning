import os
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("=" * 60)
print("KỸ THUẬT 1: HyDE")
print("=" * 60)

hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", """Hãy viết một đoạn văn bản chi tiết (khoảng 100-150 từ) trả lời câu hỏi sau,
như thể bạn đang trích xuaatss từ sổ tay nhân viên công ty.
Viết ở dạng tài liệu chính thức, không phải dạng hội thoại."""),
    ("user", "{question}"),
])
hyde_chain = hyde_prompt | llm | StrOutputParser()

def hyde_search(question, vectorstore, k=3):
    hypothetical_doc = hyde_chain.invoke({"question": question})
    print(f"Hypothetical Document:\n '{hypothetical_doc[:150]}...'\n")
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results

question = "Chính sách OT?"
print(f"Query: '{question}'\n")

print("Search thường bằng câu hỏi ngắn:")
normal_results = vectorstore.similarity_search(question, k=3)
for i, doc in enumerate(normal_results):
    print(f"[{i}] {doc.page_content[:80]}...")

print(f"\nHyDe Search (bằng hypothetical document):")
hyde_results = hyde_search(question, vectorstore, k=3)
for i, doc in enumerate(hyde_results):
    print(f"[{i}] {doc.page_content[:80]}...")

print("\n\n" + "=" * 60)
print("KỸ THUẬT 2: QUERY DECOMPOSITION")
print("=" * 60)

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia phân tích câu hỏi.
Hãy phân tách câu hỏi phức tạp thành 2-4 câu hỏi con đơn giản.
Mỗi câu hỏi con trên 1 dòng, bắt đầu bằng "- ".
Chỉ trả về danh sách câu hỏi con, không giải thích thêm."""),
    ("user", "{question}"),
])
decompose_chain = decompose_prompt | llm | StrOutputParser()

def query_decomposition_search(question, retriever, llm):
    sub_questions_text = decompose_chain.invoke({"question": question})
    sub_questions = [
        q.strip().lstrip("- ").strip()
        for q in sub_questions_text.strip().split("\n")
        if q.strip() and q.strip().startswith("-")
    ]
    
    print("Sub-questions:")
    for i, sq in enumerate(sub_questions):
        print(f"{i+1}. {sq}")

    all_contexts = []
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        context = "\n".join(d.page_content for d in docs)
        all_contexts.append(f"[Sub-Q: {sq}]\n{context}")
        print(f"\n'{sq}' -> {len(docs)} chunks retrieved")

    combined_context = "\n\n===\n\n".join(all_contexts)

    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Trả lời câu hỏi dựa trên CONTEXT. Trả lời đầy đủ, có cấu trúc."),
        ("user", "CONTEXT:\n{context}\n\nCÂU HỎI GỐC: {question}"),
    ])
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    answer = synthesis_chain.invoke({
        "context": combined_context,
        "question": question,
    })
    return answer

complex_question = "So sánh chế độ đãi ngộ giữa nhân viên thử việc và nhân viên chính thức, bao gồm lương và ngày phép?"

print(f"\nCâu hỏi phức tạp: '{complex_question}'")
answer = query_decomposition_search(complex_question, retriever, llm)
print(f"\nCâu trả lời tổng hợp:\n{answer}")

print("\n\n" + "=" * 60)
print("SO SÁNH: Normal vs HyDE vs Decomposition")
print("=" * 60)

test_q = "Fresher được đào tạo những gì và lương bao nhiêu?"

print(f"\n'{test_q}'\n")
print("Normal Search:")
for doc in vectorstore.similarity_search(test_q, k=2):
    print(f"{doc.page_content[:80]}...")

print("\nHyDe Search:")
for doc in hyde_search(test_q, vectorstore, k=2):
    print(f"{doc.page_content[:80]}...")

print("\nQuery Decomposition:")
answer = query_decomposition_search(test_q, retriever, llm)
print(f"{answer[:200]}...")