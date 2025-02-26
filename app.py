import os
import glob
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ No se encontró la clave de API de OpenAI. El servicio no funcionará correctamente.")

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

# Definir modelo para la solicitud
class QuestionRequest(BaseModel):
    question: str

# Rutas para almacenamiento
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
PDF_DIR = os.path.join(os.getcwd(), "pdf")  # Nueva carpeta para PDFs en el repositorio
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")

# Crear carpetas si no existen
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)  # Crear carpeta de PDFs si no existe
os.makedirs(CHROMA_DIR, exist_ok=True)

# Inicializar ChromaDB con manejo de errores
try:
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db.get_or_create_collection(name="lean_fabric")
    logger.info("✅ ChromaDB inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error al inicializar ChromaDB: {str(e)}")
    db = None
    collection = None

# Inicializar el modelo de lenguaje con manejo de errores
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)
    logger.info("✅ Modelo LLM inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error al inicializar el modelo LLM: {str(e)}")
    llm = None

# Función para procesar PDF y añadir a la base de datos
def process_pdf(file_path):
    try:
        # Cargar PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"📄 PDF cargado: {file_path} - {len(documents)} páginas")
        
        # Dividir en fragmentos
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_chunks = splitter.split_documents(documents)
        logger.info(f"✂️ Documento dividido en {len(docs_chunks)} fragmentos")
        
        # Almacenar en ChromaDB
        if collection:
            start_id = collection.count()
            for i, doc in enumerate(docs_chunks):
                doc_id = str(start_id + i)
                collection.add(
                    documents=[doc.page_content],
                    metadatas=[{"source": os.path.basename(file_path), "page": doc.metadata.get("page", 0)}],
                    ids=[doc_id]
                )
            logger.info(f"💾 {len(docs_chunks)} fragmentos añadidos a ChromaDB")
            return len(docs_chunks)
        else:
            logger.error("❌ ChromaDB no está disponible")
            return 0
    except Exception as e:
        logger.error(f"❌ Error procesando PDF: {str(e)}")
        return 0

# Nueva función para escanear la carpeta de PDFs y procesarlos
def scan_and_process_pdfs():
    if not collection:
        logger.error("❌ ChromaDB no está disponible para procesar los PDFs")
        return {"error": "Base de datos no disponible"}
    
    try:
        # Buscar todos los archivos PDF en la carpeta
        pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
        logger.info(f"🔍 Encontrados {len(pdf_files)} archivos PDF en la carpeta {PDF_DIR}")
        
        if not pdf_files:
            return {"message": "No se encontraron archivos PDF en la carpeta.", "processed": 0}
        
        # Obtener documentos ya procesados para evitar duplicados
        existing_docs = []
        if collection and collection.count() > 0:
            all_metadata = collection.get(include=["metadatas"])["metadatas"]
            for meta in all_metadata:
                source = meta.get("source")
                if source and source not in existing_docs:
                    existing_docs.append(source)
        
        # Procesar solo los PDFs nuevos
        total_processed = 0
        processed_files = []
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            if filename not in existing_docs:
                chunks_added = process_pdf(pdf_path)
                total_processed += chunks_added
                processed_files.append(filename)
                logger.info(f"✅ Procesado nuevo archivo: {filename}")
            else:
                logger.info(f"⏭️ Archivo ya procesado anteriormente: {filename}")
        
        return {
            "message": f"Procesados {len(processed_files)} nuevos archivos PDF",
            "processed_files": processed_files,
            "total_fragments_added": total_processed,
            "total_documents": collection.count() if collection else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error al escanear la carpeta de PDFs: {str(e)}")
        return {"error": str(e), "processed": 0}

# Función para buscar en ChromaDB
def search_chroma(query, k=3):
    if not collection:
        return [["No hay conexión a la base de datos."]]
    
    if collection.count() == 0:
        return [["No hay documentos indexados en la base de datos."]]
    
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"]

# Función para responder preguntas con GPT
def ask_ai(question):
    if not llm:
        return "El servicio de IA no está disponible en este momento."
    
    search_results = search_chroma(question, k=3)
    context = "\n".join([str(item) for sublist in search_results for item in sublist])

    prompt = (
        f"Eres un asistente util que ayuda a los estudiantes a aprender conceptos de Lean Manufacturing, con ayuda del kit del simulador: GREEN LeanMan Grand Car Factory Simulation. A continuación se presentan fragmentos de documentos sobre Lean Manufacturing:"
        f"\n\n{context}\n\n"
        f"Basado en esta información, proporciona una respuesta clara y concisa en español  si no la encuentras busca en internet información relevante que ayude a la comprensión de este kit:"
        f"\nPregunta: {question}\nRespuesta:"
    )

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"❌ Error al invocar el modelo LLM: {str(e)}")
        return f"Error al procesar la respuesta: {str(e)}"

# Endpoint para salud de la API
@app.get("/")
async def health_check():
    db_status = "disponible" if collection else "no disponible"
    llm_status = "disponible" if llm else "no disponible"
    doc_count = collection.count() if collection else 0
    
    # Contar PDFs en la carpeta del repositorio
    pdf_count = len(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    
    return {
        "status": "online",
        "database": db_status,
        "ai_model": llm_status,
        "documents_indexed": doc_count,
        "pdfs_in_repository": pdf_count
    }

# Nuevo endpoint para cargar PDFs desde la carpeta del repositorio
@app.post("/scan-repository")
async def scan_repository():
    logger.info("🔍 Iniciando escaneo de la carpeta de PDFs del repositorio")
    result = scan_and_process_pdfs()
    return result

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
        
        logger.info(f"📤 Archivo recibido: {file.filename}")
        
        # Procesar el PDF
        docs_added = process_pdf(temp_file.name)
        
        return {
            "filename": file.filename,
            "fragments_added": docs_added,
            "total_documents": collection.count() if collection else 0
        }
    except Exception as e:
        logger.error(f"❌ Error al procesar el archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        # Eliminar archivo temporal
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Endpoint para hacer preguntas
@app.post("/ask")
async def get_answer(data: QuestionRequest):
    try:
        question = data.question
        logger.info(f"❓ Pregunta recibida: {question}")
        
        if not question:
            raise HTTPException(status_code=400, detail="🚨 No se proporcionó ninguna pregunta.")
        
        response = ask_ai(question)
        logger.info("✅ Respuesta generada correctamente")
        
        return {"respuesta": response}
    
    except Exception as e:
        logger.error(f"❌ Error al procesar la pregunta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para listar documentos
@app.get("/documents")
async def list_documents():
    if not collection:
        return {"error": "Base de datos no disponible", "documents": []}
    
    if collection.count() == 0:
        return {"documents": []}
    
    try:
        all_metadata = collection.get(include=["metadatas"])["metadatas"]
        unique_sources = {}
        
        for meta in all_metadata:
            source = meta.get("source")
            if source and source not in unique_sources:
                unique_sources[source] = True
        
        return {"documents": list(unique_sources.keys())}
    except Exception as e:
        logger.error(f"❌ Error al listar documentos: {str(e)}")
        return {"error": str(e), "documents": []}

# Para testing local
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Iniciando servidor en puerto {port}")
    # Escanear PDFs al iniciar la aplicación (opcional)
    scan_and_process_pdfs()
    uvicorn.run(app, host="0.0.0.0", port=port)
