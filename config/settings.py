import yaml
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

from src.llm.enums import LLMProvider


# --- 1. Environment Settings (Secrets & Infrastructure) ---
# These come from .env file. This is the SINGLE SOURCE OF TRUTH for provider selection.
class EnvSettings(BaseSettings):
    # Neo4j
    neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(validation_alias="NEO4J_PASSWORD")

    # LLM Provider Selection (single source of truth)
    llm_provider: str = Field("gemini", validation_alias="LLM_PROVIDER")
    
    # Ollama
    ollama_base_url: str = Field("http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("gemma3:4b", validation_alias="OLLAMA_MODEL")
    
    # Gemini
    gemini_api_key: Optional[str] = Field(None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-3-flash-preview", validation_alias="GEMINI_MODEL")
    
    # vLLM
    vllm_base_url: str = Field("http://localhost:8000/v1", validation_alias="VLLM_BASE_URL")
    vllm_api_key: str = Field("EMPTY", validation_alias="VLLM_API_KEY")
    vllm_model: str = Field("meta-llama/Meta-Llama-3-8B-Instruct", validation_alias="VLLM_MODEL")

    # LangSmith
    langchain_tracing_v2: bool = Field(False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("project-graph-rag", validation_alias="LANGCHAIN_PROJECT")
    
    # Feature Flags
    enable_guardrail: bool = Field(True, validation_alias="ENABLE_GUARDRAIL")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# --- 2. YAML Config Models (Application Logic) ---
class LLMConfig(BaseModel):
    temperature: float = 0.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class GraphConfig(BaseModel):
    chunk_size: int = 2000
    chunk_overlap: int = 500


class GeneralConfig(BaseModel):
    llm: LLMConfig
    graph: GraphConfig


# --- 3. Prompt Models (Text Templates) ---
class PromptsConfig(BaseModel):
    extraction_prompt: str
    rag_prompt: str


# --- 4. Master Settings Class ---
class Settings(BaseModel):
    env: EnvSettings
    general: GeneralConfig
    prompts: PromptsConfig


def load_settings() -> Settings:
    # 1. Load Env
    env_settings = EnvSettings()

    # 2. Load General YAML
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    general_path = os.path.join(base_dir, "config", "general_config.yaml")

    with open(general_path, "r") as f:
        general_dict = yaml.safe_load(f)
    general_config = GeneralConfig(**general_dict)

    # 3. Load Prompts YAML
    prompts_path = os.path.join(base_dir, "config", "prompts.yaml")
    with open(prompts_path, "r") as f:
        prompts_dict = yaml.safe_load(f)
    prompts_config = PromptsConfig(**prompts_dict)

    return Settings(env=env_settings, general=general_config, prompts=prompts_config)


# Singleton instance
settings = load_settings()
