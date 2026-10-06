"""Central configuration layer for AgentDesk.

All environment access happens here. The rest of the application must use
`get_config()` instead of calling os.getenv() directly. Secret values are
never logged.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_QDRANT_URL = "http://localhost:6333"


def groq_litellm_model(groq_model: str) -> str:
    """Prefix a Groq model id with LiteLLM's Groq provider prefix.

    Keeps routing explicit: CrewAI -> LiteLLM -> Groq API, never OpenAI,
    even when the model id itself contains a slash (e.g. openai/gpt-oss-120b).
    """
    if not groq_model:
        raise ValueError("groq_model must be a non-empty string")
    if groq_model.startswith("groq/"):
        return groq_model
    return f"groq/{groq_model}"


@dataclass(frozen=True)
class Config:
    groq_api_key: str = field(repr=False)
    groq_model: str
    qdrant_url: str
    tavily_api_key: str = field(repr=False)

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_config() -> Config:
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        qdrant_url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
    )
