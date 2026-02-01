from langchain_core.prompts import PromptTemplate
from src.llm import get_llm_client, get_graph_client
from src.llm.graph_client import GraphManager
from src.llm.hybrid_search import HybridRetriever
from config.settings import settings

def chat_loop():
    print("Initializing Chatbot (Hybrid RAG)...")
    
    # 1. Initialize LLM (supports vLLM if configured)
    llm = get_llm_client()
    
    # 2. Initialize Graph Manager & Hybrid Retriever
    # Note: get_graph_client returns Neo4jGraph, but GraphManager wraps it with our logic.
    # We should instantiate GraphManager directly or refactor get_graph_client.
    # For now, let's just make a new GraphManager instance as it is lightweight.
    graph_manager = GraphManager()
    retriever = HybridRetriever(graph_manager)
    
    # 3. Setup Prompt
    template = """You are a helpful assistant for the UET Lahore Prospectus.
    Use the following pieces of context to answer the user's question. 
    The context includes relevant document excerpts and details about related entities (Departments, Persons, etc.).
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    chain = prompt | llm

    print("Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        try:
            print("  Searching knowledge base...")
            # 1. Retrieve Hybrid Context (Vector + Graph)
            context = retriever.search(query, top_k=3)
            
            if not context:
                print("Bot: I couldn't find any relevant information in the documents.")
                continue
                
            # 2. Generate Answer
            response = chain.invoke({"context": context, "question": query})
            
            # response is AIMessage if using ChatModel
            content = response.content if hasattr(response, "content") else response
            print(f"Bot: {content}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    chat_loop()
