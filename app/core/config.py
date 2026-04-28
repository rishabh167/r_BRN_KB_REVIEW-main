import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "3306")
    DB_NAME: str = os.getenv("DB_NAME", "brn_admin_panel")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    LITELLM_API_BASE: str = os.getenv("LITELLM_API_BASE", "http://localhost:4000/v1")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")
    FIREWORKS_API_BASE: str = os.getenv("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1")
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Direct provider keys (also consumed by LiteLLM if running as proxy)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8062"))
    APP_ENV: str = os.getenv("APP_ENV", "dev")

    X_API_KEY: str = os.getenv("X_API_KEY", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def DB_URI(self) -> str:
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"mysql+mysqlconnector://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def get_provider_key(self, provider: str) -> str:
        mapping = {
            "litellm": self.LITELLM_API_KEY,
            "fireworks": self.FIREWORKS_API_KEY,
            "openrouter": self.OPENROUTER_API_KEY,
            # Direct provider keys
            "anthropic": self.ANTHROPIC_API_KEY,
            "google": self.GOOGLE_API_KEY,
        }
        return mapping.get(provider, "")

    def get_provider_base(self, provider: str) -> str:
        mapping = {
            "litellm": self.LITELLM_API_BASE,
            "fireworks": self.FIREWORKS_API_BASE,
            "openrouter": self.OPENROUTER_API_BASE,
            # Google Gemini exposes an OpenAI-compatible endpoint natively
            "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
            # Anthropic has partial OpenAI-compat but requires extra headers;
            # recommended: route through litellm or openrouter instead.
            "anthropic": "https://api.anthropic.com/v1",
        }
        return mapping.get(provider, "")



settings = Settings()
