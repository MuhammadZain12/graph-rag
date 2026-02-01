from langchain_ollama import ChatOllama
from config.settings import settings

def get_ollama_client():
    return ChatOllama(
        model=settings.env.ollama_model,
        temperature=settings.general.llm.temperature,
        base_url=settings.env.ollama_base_url
    )
