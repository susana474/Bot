import os
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import shutil

# Configuración de API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 No se encontró la clave de API de OpenAI.")

# Inicializar FastAPI
app = FastAPI(title="Lean Manufacturing Q&A")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpetas para almacenamiento
UPLOAD_FOLDER = "./uploads"
CHROMA_PATH = "./chroma_db"

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# Variables globales
db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db.get_or_create_collection(name="lean_fabric")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# Verificar si hay documentos indexados al inicio
docs_count = len(collection.get(include=[])["ids"]) if collection.count() > 0 else 0
print(f"ℹ️ {docs_count} documentos encontrados en la base de datos")

# Función para procesar PDF y añadir a la base de datos
def process_pdf(file_path):
    try:
        # Cargar PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Dividir en fragmentos
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_chunks = splitter.split_documents(documents)
        
        # Almacenar en ChromaDB
        start_id = collection.count()
        for i, doc in enumerate(docs_chunks):
            doc_id = str(start_id + i)
            collection.add(
                documents=[doc.page_content],
                metadatas=[{"source": os.path.basename(file_path), "page": doc.metadata.get("page", 0)}],
                ids=[doc_id]
            )
        
        return len(docs_chunks)
    except Exception as e:
        print(f"Error procesando PDF: {str(e)}")
        return 0

# Función para buscar en ChromaDB
def search_chroma(query, k=3):
    if collection.count() == 0:
        return [["No hay documentos indexados en la base de datos."]]
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"]

# Función para responder preguntas con GPT
def ask_ai(question):
    search_results = search_chroma(question, k=3)
    context = "\n".join([str(item) for sublist in search_results for item in sublist])

    prompt = (
        f"A continuación se presentan fragmentos de documentos sobre Lean Manufacturing:"
        f"\n\n{context}\n\n"
        f"Basado en esta información, proporciona una respuesta clara y concisa en español:"
        f"\nPregunta: {question}\nRespuesta:"
    )

    response = llm.invoke(prompt)
    return response

# Endpoint para salud de la API
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "documents_indexed": collection.count()
    }

# Endpoint para subir PDFs
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
    
    # Guardar el archivo temporalmente
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        contents = await file.read()
        with open(temp_file.name, "wb") as f:
            f.write(contents)
        
        # Procesar el PDF
        docs_added = process_pdf(temp_file.name)
        
        return {
            "filename": file.filename,
            "fragments_added": docs_added,
            "total_documents": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        # Eliminar archivo temporal
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Endpoint para hacer preguntas
@app.post("/ask")
async def get_answer(data: dict):
    try:
        question = data.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="🚨 No se proporcionó ninguna pregunta.")
        
        response = ask_ai(question)
        return {"respuesta": str(response.content)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para listar documentos
@app.get("/documents")
async def list_documents():
    if collection.count() == 0:
        return {"documents": []}
    
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    unique_sources = {}
    
    for meta in all_metadata:
        source = meta.get("source")
        if source and source not in unique_sources:
            unique_sources[source] = True
    
    return {"documents": list(unique_sources.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
