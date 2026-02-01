from langchain_neo4j import Neo4jGraph
from config.settings import settings

def get_graph_client():
    return Neo4jGraph(
        url=settings.env.neo4j_uri,
        username=settings.env.neo4j_username,
        password=settings.env.neo4j_password
    )

class GraphManager:
    def __init__(self):
        self.driver = Neo4jGraph(
            url=settings.env.neo4j_uri,
            username=settings.env.neo4j_username,
            password=settings.env.neo4j_password
        )

    def add_chunk(self, chunk_id: str, text: str, embedding: list = None):
        """
        Creates a Chunk node with the full text content and optional embedding.
        """
        try:
            cypher = "MERGE (c:Chunk {id: $id}) SET c.text = $text"
            params = {"id": chunk_id, "text": text}
            
            if embedding:
                cypher += ", c.embedding = $embedding"
                params["embedding"] = embedding
                
            self.driver.query(cypher, params)
        except Exception as e:
            print(f"Error adding chunk {chunk_id}: {e}")

    def create_vector_index(self, dimension: int = 384):
        """
        Creates a vector index on Chunk nodes if it doesn't exist.
        """
        try:
            # Check if index exists (Neo4j 5.x+)
            # Note: Syntax varies by version. Using declarative syntax.
            cypher = f"""
            CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            self.driver.query(cypher)
            print("Vector index 'chunk_vector_index' ensured.")
        except Exception as e:
            print(f"Error creating vector index: {e}")

    def query_vector_index(self, query_embedding: list, top_k: int = 5) -> list:
        """
        Queries the vector index for similar chunks.
        """
        try:
            cypher = """
            CALL db.index.vector.queryNodes('chunk_vector_index', $k, $embedding)
            YIELD node, score
            RETURN node.text AS text, node.id AS id, score
            """
            result = self.driver.query(cypher, {"k": top_k, "embedding": query_embedding})
            return result
        except Exception as e:
            print(f"Error querying vector index: {e}")
            return []

    def add_graph_data(self, data: dict, chunk_id: str = None):
        """
        Ingests data dictionary with 'nodes' and 'edges'.
        """
        # 1. Merge Nodes
        nodes = data.get("nodes", []) or data.get("entities", [])
        for node in nodes:
            try:
                # Sanitize type
                label = "".join(c for c in node['type'] if c.isalnum() or c == '_')
                if not label: label = "Entity"
                
                # Prepare properties for Cypher
                props = node.get("properties", {})
                props["id"] = node["id"]
                props["name"] = node.get("name", "Unknown") # Ensure name is set
                
                # Cypher query to merge the node
                # We simply set all properties found in the dict
                cypher = f"MERGE (e:`{label}` {{id: $id}}) SET e += $props"
                self.driver.query(cypher, {"id": node['id'], "props": props})

                # Link to Chunk
                if chunk_id:
                    link_cypher = f"""
                    MATCH (e:`{label}` {{id: $id}})
                    MATCH (c:Chunk {{id: $chunk_id}})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """
                    self.driver.query(link_cypher, {"id": node['id'], "chunk_id": chunk_id})

            except Exception as e:
                print(f"Error adding node {node.get('id')}: {e}")

        # 2. Merge Edges
        edges = data.get("edges", []) or data.get("relationships", [])
        for edge in edges:
            try:
                # Sanitize relationship type
                rel_type = edge['type'].replace(" ", "_").upper()
                rel_type = "".join(c for c in rel_type if c.isalnum() or c == '_')
                if not rel_type: rel_type = "RELATED_TO"
                
                edge_props = edge.get("properties", {})

                # Cypher to merge relationship
                cypher = f"""
                MATCH (a {{id: $source}})
                MATCH (b {{id: $target}})
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r += $props
                """
                self.driver.query(cypher, {
                    "source": edge['source'], 
                    "target": edge['target'],
                    "props": edge_props
                })
            except Exception as e:
                print(f"Error adding edge {edge}: {e}")
