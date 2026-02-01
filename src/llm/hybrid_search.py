from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from src.llm.graph_client import GraphManager

class HybridRetriever:
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.general.llm.embedding_model
        )

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Performs hybrid search:
        1. Vector search for relevant chunks.
        2. Retrieves entities mentioned in those chunks.
        3. Formats context.
        """
        # 1. Vector Search
        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.graph_manager.query_vector_index(query_embedding, top_k=top_k)
        
        if not vector_results:
            return ""

        context_parts = []
        chunk_ids = []

        # 2. Process Chunks
        for result in vector_results:
            text = result['text']
            chunk_id = result['id']
            score = result['score']
            chunk_ids.append(chunk_id)
            context_parts.append(f"--- Document Chunk (ID: {chunk_id}, Score: {score:.2f}) ---\n{text}")

        # 3. Graph Enrichment (Get entities linked to these chunks)
        # We find entities that are MENTIONED_IN these chunks
        if chunk_ids:
            cypher = """
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            WHERE c.id IN $chunk_ids
            RETURN e.name AS name, e.type AS type, e.description AS description, labels(e) AS labels
            """
            # Note: e.description might not exist, but we get properties.
            # Let's just return the whole node property map is better but Neo4j returns it as dict.
            
            # Revised Cypher to request specific properties or map
            cypher = """
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            WHERE c.id IN $chunk_ids
            RETURN DISTINCT e
            """
            
            try:
                # We need to access the driver directly or add a method.
                # GraphManager exposes .driver (Neo4jGraph).
                # Neo4jGraph.query returns a list of dictionaries.
                entity_results = self.graph_manager.driver.query(cypher, {"chunk_ids": chunk_ids})
                
                if entity_results:
                    context_parts.append("\n--- Key Entities Mentioned in Context ---")
                    seen_entities = set()
                    for record in entity_results:
                        # record['e'] is the node dict properties
                        node = record['e']
                        # Try to get meaningful name
                        name = node.get('name') or node.get('id')
                        # Try to infer label/type (Neo4jGraph result might not include labels in the node dict easily unless we return labels(e))
                        
                        # Let's simplify the display
                        if name and name not in seen_entities:
                            props_str = ", ".join([f"{k}: {v}" for k, v in node.items() if k not in ['embedding', 'text', 'id']])
                            context_parts.append(f"Entity: {name} | Details: {props_str}")
                            seen_entities.add(name)

            except Exception as e:
                print(f"Error retrieving linked entities: {e}")

        return "\n\n".join(context_parts)
