import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)

embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"}
)

loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Dựa trên lịch sử hội thoại, hãy viết lại câu hỏi mới nhất thành câu hỏi độc lập (standalone question) để tìm kiếm tài liệu
Nếu câu hỏi đã rõ ràng, giữ nguyên"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI. Trả lời dựa trên CONTEXT bên dưới.
QUY TẮC:
1. CHỈ dùng thông tin từ CONTEXT.
2. Nếu không có thông tin, nói rõ.
3. Cuối câu trả lời, ghi [Nguồn: tên section trong tài liệu].
"""),
    ("user", """CONTEXT:
{context}

CÂU HỎI: {question}
"""),
])

answer_chain = rag_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

print("RAG Chatbot - Hỏi đáp về quy định công ty ABC")
print("Gõ 'quit' để thoát\n")

chat_history = []

while True:
    question = input("Bạn hỏi: ")
    if question.lower() in ["quit", "exit"]:
        print("Tạm biệt!")
        break

    if chat_history:
        standalone_q = rewrite_chain.invoke({
            "chat_history": chat_history,
            "question": question,
        })
        print(f"Câu hỏi rewritten: {standalone_q}")
    else:
        standalone_q = question

    relevant_docs = retriever.invoke(standalone_q)

    context = format_docs(relevant_docs)
    
    answer = answer_chain.invoke({
        "context": context,
        "question": standalone_q
    })
    print(f"\n{answer}\n")
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))