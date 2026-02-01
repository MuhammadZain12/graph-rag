from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

def get_gemini_client():
    return ChatGoogleGenerativeAI(
        model=settings.env.gemini_model,
        temperature=settings.general.llm.temperature,
        google_api_key=settings.env.gemini_api_key,
        request_timeout=60,
        max_retries=2
    )
