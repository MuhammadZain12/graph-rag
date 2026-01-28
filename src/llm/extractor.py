from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
from src.llm import get_llm_client
from src.prompt_engineering.extraction import GraphData
from config.settings import settings

from .utils import retry_with_backoff

def get_extraction_chain():
    # Use extraction_provider if configured, otherwise fallback to default logic
    provider = getattr(settings.general.llm, "extraction_provider", None) 
    llm = get_llm_client(provider)
    # Use prompt from config
    prompt_str = settings.prompts.extraction_prompt
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_str),
        ("human", "Text chunk: {text}")
    ])
    
    # Enforce strict schema using with_structured_output
    structured_llm = llm.with_structured_output(GraphData)

    chain = prompt | structured_llm
    return chain

@retry_with_backoff(retries=3, initial_delay=2.0)
def extract_graph_from_text(text: str) -> dict:
    chain = get_extraction_chain()
    # Let decorator handle the exception for retries
    result = chain.invoke({"text": text})
    return result.model_dump()
