from langchain_openai import ChatOpenAI
from config.settings import settings

def get_vllm_client():
    """
    Returns a configured ChatOpenAI client pointing to a vLLM server.
    vLLM provides an OpenAI-compatible API.
    """
    return ChatOpenAI(
        model=settings.env.vllm_model,
        temperature=settings.general.llm.temperature,
        openai_api_key=settings.env.vllm_api_key,
        openai_api_base=settings.env.vllm_base_url,
    )
