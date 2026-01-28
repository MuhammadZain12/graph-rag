from .ollama_client import get_ollama_client
from .gemini_client import get_gemini_client
from config.settings import settings
from .enums import LLMProvider

def get_llm_client(provider: str = None):
    # Determine provider: argument > env settings > general config > default
    if provider is None:
        if settings.env.llm_provider:
             provider = settings.env.llm_provider
        elif hasattr(settings.general.llm, "provider"):
             provider = settings.general.llm.provider
        else:
            provider = LLMProvider.OLLAMA.value # Default fallback
            
    if provider == LLMProvider.GEMINI.value:
        return get_gemini_client()
    elif provider == LLMProvider.OLLAMA.value:
        return get_ollama_client()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
