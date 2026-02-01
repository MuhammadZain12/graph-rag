import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from src.llm.extractor import extract_graph_from_text
from src.llm.graph_client import GraphManager
from src.llm import get_llm_client
from config.settings import settings

def ingest(pdf_path="data/files/UET lahore Document.pdf"):
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Initializing Graph Manager...")
    graph_manager = GraphManager()
    
    # Initialize Embedding Model
    print(f"Loading Embedding Model: {settings.general.llm.embedding_model}...")
    embedding_model = HuggingFaceEmbeddings(model_name=settings.general.llm.embedding_model)
    
    # Ensure Vector Index Exists
    graph_manager.create_vector_index()

    # Optimize chunk size for Gemini
    chunk_size = settings.general.graph.chunk_size
    chunk_overlap = settings.general.graph.chunk_overlap
    
    # Use extraction provider for logic
    provider = getattr(settings.general.llm, "extraction_provider", "ollama")
    print(f"Using LLM Provider: {provider}")
    
    if provider == "gemini":
        print("Optimizing chunk size for Gemini (2500 chars, 500 overlap)...")
        chunk_size = 2500
        chunk_overlap = 500
    
    print(f"Splitting {len(documents)} pages with chunk_size={chunk_size}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"Processing {len(texts)} chunks...")
    for i, doc in enumerate(texts):
        print(f"  - Chunk {i+1}/{len(texts)}")
        try:
            # 1. Generate unique Chunk ID
            chunk_id = str(uuid.uuid4())
            chunk_text = doc.page_content
            
            # 2. Store Chunk Text & Embedding (Hybrid RAG)
            embedding = embedding_model.embed_query(chunk_text)
            graph_manager.add_chunk(chunk_id, chunk_text, embedding=embedding)

            # 3. Extract Graph Data using LLM
            print(f"    Extracting from chunk {chunk_id[:8]}...")
            graph_data = extract_graph_from_text(chunk_text)
            print("    Extraction complete.")
            
            # 4. Ingest Graph Data & Link to Chunk
            if graph_data:
                graph_manager.add_graph_data(graph_data, chunk_id=chunk_id)
                
        except Exception as e:
            print(f"Failed to process chunk {i+1}: {e}")

    print("Ingestion Complete!")

if __name__ == "__main__":
    ingest()
