from langchain_ollama import ChatOllama
from config.settings import settings


def get_ollama_client():
    return ChatOllama(
        model=settings.general.llm.model_name,
        base_url=settings.env.ollama_base_url,
        temperature=settings.general.llm.temperature,
    )
