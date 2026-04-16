import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

print("Step 1: Loading document ...")
loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()
print(f"Đã load {len(documents)} tài liệu")
print("="*50 + "\n")

print("Step 2: Chunking ...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Đã cắt thành {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content[:60]}...")
print("="*50 + "\n")

print("\nStep 3: Embedding & Indexing...")
embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"Đã index {len(chunks)} chunks vào FAISS")
print("="*50 + "\n")

print("\nStep 4: Create Retriever...")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
print("="*50 + "\n")

print("\nStep 5: Create RAG chain...\n")
llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

QUY TẮC:
1. Chỉ trả lời dựa trên thông tin trong CONTEXT bên dưới.
2. Nếu CONTEXT không chứa thông tin cần thiết, nói "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trích dẫn cụ thể từ tài liệu khi trả lời.
"""),
    ("user", """CONTEXT (tài liệu tham khảo):
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""),
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

questions = [
    "Nhân viên chính thức được bao nhiêu ngày phép một năm?",
    "Lương OT ngày lễ tính bao nhiêu phần trăm?",
    "Budget đào tạo cho mỗi nhân viên là bao nhiêu?",
    "Thưởng tháng 13 được trả khi nào?",
]

for q in questions:
    print(f"Question: {q}\n")
    answer = rag_chain.invoke(q)
    print(f"{answer}\n")
    print("-"*50)