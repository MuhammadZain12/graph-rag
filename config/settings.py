import yaml
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

from src.llm.enums import LLMProvider


# --- 1. YAML Config Models (Logic) ---
class LLMConfig(BaseModel):
    model_name: str
    temperature: float
    provider: str = LLMProvider.OLLAMA.value  # default provider
    extraction_provider: str = LLMProvider.GEMINI.value  # specific for extraction
    gemini_model: str = "gemini-2.5-flash"


class GraphConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class GeneralConfig(BaseModel):
    llm: LLMConfig
    graph: GraphConfig


# --- 2. Prompt Models (Text) ---
class PromptsConfig(BaseModel):
    extraction_prompt: str
    rag_prompt: str


# --- 3. Env Settings (Secrets/Infra) ---
class EnvSettings(BaseSettings):
    neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_username: str = Field(validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(validation_alias="NEO4J_PASSWORD")

    ollama_base_url: str = Field(
        "http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    gemini_api_key: Optional[str] = Field(None, validation_alias="GEMINI_API_KEY")
    llm_provider: str = Field("ollama", validation_alias="LLM_PROVIDER")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# --- 4. Master Settings Class ---
class Settings(BaseModel):
    # Composition of all configs
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
