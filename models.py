from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class DocumentMetadata(BaseModel):
    filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class QAConfig(BaseModel):
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

class QuestionRequest(BaseModel):
    question: str
    document_id: str

class SourceDocument(BaseModel):
    content: str
    metadata: Dict
    
class QuestionResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    metadata: Dict
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str
    status: str