from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(config):
        if config.embeddings_type == "openai":
            return OpenAIEmbeddings(**config.embeddings_config or {})
        elif config.embeddings_type == "ollama":
            return OllamaEmbeddings(**config.embeddings_config or {})
        else:
            raise ValueError(f"Unsupported embeddings type: {config.embeddings_type}")