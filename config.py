from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class QAConfig:
    source_file_path: str = "source_data.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_type: str = "chroma"
    vector_store_path: str = "db"
    llm_type: str = "openai"
    llm_config: Dict[str, Any] = None
    embeddings_type: str = "openai"
    embeddings_config: Dict[str, Any] = None
    prompt_template: str = ""

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]

@dataclass
class QAResponse:
    answer: str
    source_documents: List[Document]
    metadata: Dict[str, Any]
