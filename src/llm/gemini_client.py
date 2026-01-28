from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
import os

def get_gemini_client():
    # Ensure API key is set either in env vars or settings
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and hasattr(settings.env, "gemini_api_key"):
        api_key = settings.env.gemini_api_key
        
    return ChatGoogleGenerativeAI(
        model=settings.general.llm.gemini_model,
        temperature=settings.general.llm.temperature,
        google_api_key=api_key,
        request_timeout=60,
        max_retries=2
    )
