from .ollama_client import get_ollama_client
from .gemini_client import get_gemini_client
from .vllm_client import get_vllm_client
from config.settings import settings
from .enums import LLMProvider

def get_llm_client(provider: str = None):
    """
    Get LLM client based on provider.
    
    Priority: argument > LLM_PROVIDER env var
    """
    if provider is None:
        provider = settings.env.llm_provider
            
    if provider == LLMProvider.GEMINI.value:
        return get_gemini_client()
    elif provider == LLMProvider.OLLAMA.value:
        return get_ollama_client()
    elif provider == LLMProvider.VLLM.value:
        return get_vllm_client()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Valid options: ollama, gemini, vllm")
