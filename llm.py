from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama

from config import QAConfig

class LLMFactory:
    @staticmethod
    def create_llm(config: QAConfig):
        if config.llm_type == "openai":
            return ChatOpenAI(**config.llm_config or {})
        elif config.llm_type == "ollama":
            return Ollama(**config.llm_config or {})
        else:
            raise ValueError(f"Unsupported LLM type: {config.llm_type}")