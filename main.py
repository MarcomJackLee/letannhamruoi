from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import gradio as gr
import os

# === 3. KHỞI TẠO FASTAPI ===
app = FastAPI()

# === 4. CLASS CHO API ===
class UserQuery(BaseModel):
    question: str

# === 5. LOAD TÀI LIỆU ===
DOCUMENTS_DIR = "./docs"

def load_documents():
    documents = []
    for file in os.listdir(DOCUMENTS_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(DOCUMENTS_DIR, file))
            documents.extend(loader.load())
    return documents

# === 6. TẠO VECTORSTORE ===
def create_vectorstore():
    docs = load_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

# === 7. LOAD VECTORSTORE (DÙNG SAU KHI ĐÃ TẠO) ===
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(persist_directory="db", embedding_function=embeddings)

# === 8. TẠO AI LỄ TÂN ===
retriever = load_vectorstore().as_retriever()
llm = Ollama(model="mistral")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# === 9. API CHO WEB ===
@app.post("/ask")
async def ask_question(query: UserQuery):
    result = qa_chain.invoke({"question": query.question})
    return {"answer": result["answer"]}

# === 10. GIAO DIỆN GRADIO ===
def chat_interface(message, history=[]):
    result = qa_chain.invoke({"question": message})
    return result["answer"]

ui = gr.ChatInterface(fn=chat_interface, title="Lễ Tân AI Khách Sạn", examples=["Khách sạn có bữa sáng không?", "Có chỗ đậu xe không?"], chatbot=gr.Chatbot())

# === 11. CHẠY GRADIO ===
if __name__ == "__main__":
    ui.launch(server_port=7860)