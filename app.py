import os
import chromadb
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from getpass import getpass
import openai

if "OPENAI_API_KEY" not in os.environ:
   openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar FastAPI
app = FastAPI()

# 📂 Carpeta donde están los PDFs públicos de Google Drive
PDF_FOLDER = "https://drive.google.com/drive/folders/1bgXRNCMuPAX6JCizDGovpAZplPFjub9N?usp=sharing"

# Buscar PDFs en la carpeta local
pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

# Cargar PDFs en LangChain
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

# Dividir en fragmentos de 500 caracteres
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_chunks = splitter.split_documents(documents)

# Inicializar modelo de IA (GPT-3.5)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# Conectar a ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="lean_fabric")

# Almacenar fragmentos en ChromaDB
for i, doc in enumerate(docs_chunks):
    collection.add(
        documents=[doc.page_content],
        metadatas=[{"source": f"fragment_{i}"}],
        ids=[str(i)]
    )

print(f"✅ {len(documents)} documentos indexados en ChromaDB")

# 🔍 Función para buscar en ChromaDB
def search_chroma(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"]

# 🎯 Función para responder preguntas usando GPT
def ask_ai(question):
    search_results = search_chroma(question, k=3)
    context = "\n".join([str(item) for sublist in search_results for item in sublist])

    prompt = (f"A continuación se presentan fragmentos de documentos sobre Lean Manufacturing:"
              f"\n\n{context}\n\n"
              f"Basado en esta información, proporciona una respuesta clara y concisa en español:"
              f"\nPregunta: {question}\nRespuesta:")

    response = llm.invoke(prompt)
    return response


# 📌 API para recibir preguntas
@app.get("/ask")
def get_answer(question: str):
    try:
        response = ask_ai(question)
        return {"respuesta": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
