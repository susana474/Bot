import os
import chromadb
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 📌 Asegurar clave API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 No se encontró la clave de API de OpenAI.")

# Inicializar FastAPI
app = FastAPI()

# 📂 Leer PDFs desde la misma carpeta donde está app.py
PDF_FOLDER = "."

# 📌 Buscar TODOS los PDFs en la carpeta raíz
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print("⚠️ No se encontraron PDFs en la carpeta.")
else:
    print(f"📄 Archivos PDF encontrados: {pdf_files}")

# 📌 Cargar PDFs en LangChain
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(os.path.join(PDF_FOLDER, pdf))
    documents.extend(loader.load())

# 📌 Dividir en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_chunks = splitter.split_documents(documents)

# 📌 Inicializar modelo de IA (GPT-3.5)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# 📌 Conectar a ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="lean_fabric")

# 📌 Almacenar fragmentos en ChromaDB
for i, doc in enumerate(docs_chunks):
    collection.add(
        documents=[doc.page_content],
        metadatas=[{"source": f"fragment_{i}"}],
        ids=[str(i)]
    )

print(f"✅ {len(documents)} documentos indexados en ChromaDB")

# 📌 API para recibir preguntas y responder con IA
@app.post("/ask")
async def get_answer(data: dict):
    try:
        question = data.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="🚨 No se proporcionó ninguna pregunta.")
        
        response = llm.invoke(question)
        return {"respuesta": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 6000)))

