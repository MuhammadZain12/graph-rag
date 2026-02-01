"""
Script to add embeddings to existing Chunk nodes in Neo4j.
This updates chunks that were ingested before embedding support was added.
"""
import sys
sys.path.insert(0, ".")

from langchain_huggingface import HuggingFaceEmbeddings
from src.llm.graph_client import GraphManager
from config.settings import settings


def update_embeddings():
    print("=" * 50)
    print("Embedding Update Script")
    print("=" * 50)
    
    # 1. Initialize
    print(f"\n[1/4] Loading embedding model: {settings.general.llm.embedding_model}")
    embedding_model = HuggingFaceEmbeddings(model_name=settings.general.llm.embedding_model)
    
    print("[2/4] Connecting to Neo4j...")
    graph_manager = GraphManager()
    
    # 2. Create vector index if missing
    print("[3/4] Ensuring vector index exists...")
    graph_manager.create_vector_index()
    
    # 3. Fetch all Chunk nodes without embeddings
    print("[4/4] Fetching chunks without embeddings...")
    cypher = """
    MATCH (c:Chunk)
    WHERE c.embedding IS NULL
    RETURN c.id AS id, c.text AS text
    """
    chunks = graph_manager.driver.query(cypher)
    
    if not chunks:
        print("\n✅ All chunks already have embeddings. Nothing to update.")
        return
    
    print(f"\nFound {len(chunks)} chunks without embeddings. Updating...")
    
    # 4. Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['id']
        chunk_text = chunk['text']
        
        if not chunk_text:
            print(f"  [{i+1}/{len(chunks)}] Skipping chunk {chunk_id[:8]}... (no text)")
            continue
            
        try:
            # Compute embedding
            embedding = embedding_model.embed_query(chunk_text)
            
            # Update chunk in Neo4j
            update_cypher = """
            MATCH (c:Chunk {id: $id})
            SET c.embedding = $embedding
            """
            graph_manager.driver.query(update_cypher, {"id": chunk_id, "embedding": embedding})
            
            print(f"  [{i+1}/{len(chunks)}] Updated chunk {chunk_id[:8]}...")
            
        except Exception as e:
            print(f"  [{i+1}/{len(chunks)}] Error updating {chunk_id[:8]}: {e}")
    
    print("\n✅ Embedding update complete!")


if __name__ == "__main__":
    update_embeddings()
