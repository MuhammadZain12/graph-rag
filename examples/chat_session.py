from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from src.llm import get_llm_client, get_graph_client
from config.settings import settings

def chat_loop():
    print("Initializing Chatbot...")
    llm = get_llm_client()
    graph = get_graph_client()
    
    # We use Cypher chain to retrieve data
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    
    raw_prompt = settings.prompts.rag_prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=raw_prompt
    )
    
    # Only if your version of GraphCypherQAChain supports qa_prompt. 
    # Standard GraphCypherQAChain uses 'qa_prompt' for the final answer generation.
    
    # Re-initialize with custom prompt if needed, or just rely on the default + wrapper
    # For now, let's try strict prompt capability.
    
    print("Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        try:
            # Check for department context (simple keyword heuristic or LLM based)
            # For now, rely on the prompt to filter out-of-scope.
            
            response = chain.invoke({"query": query})
            print(f"Bot: {response['result']}")
        except Exception as e:
            print(f"Error: {e}")
            print("Bot: I encountered an error accessing the department information.")

if __name__ == "__main__":
    chat_loop()
