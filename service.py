import tempfile
import os
from typing import Dict, Optional

import shutil

from models import *
from fastapi import HTTPException, status
from qa_system import QASystem

current_directory = os.path.dirname(os.path.abspath(__file__))
class QAService:
    def __init__(self):
        self.documents: Dict[str, DocumentMetadata] = {}
        self.qa_systems: Dict[str, QASystem] = {}
        self.base_vector_store_path = "vector_stores"
        self.file_uploads_directory = os.path.join(current_directory, "_api_file_uploads")
        os.makedirs(self.file_uploads_directory, exist_ok=True)

    def _get_vector_store_path(self, document_id: str) -> str:
        return os.path.join(self.base_vector_store_path, document_id)

    async def process_document(self, file_content: bytes, filename: str) -> DocumentMetadata:
        metadata = DocumentMetadata(filename=filename)
    
        file_path = os.path.join(self.file_uploads_directory, filename)
        print(file_path)

        with open(file_path, 'wb') as f:
            f.write(file_content)

        system_config = QAConfig(
            vector_store_type="chroma",
            llm_type="openai",
            llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY"), "temperature": 0.8, "model": "gpt-4o-mini"},
            embeddings_type="openai",
            embeddings_config={"openai_api_key": os.getenv("OPENAI_API_KEY")}
        )

        qa_system = QASystem(system_config)
        qa_system.initialize(file_path)
        
        self.documents[metadata.document_id] = metadata
        self.qa_systems[metadata.document_id] = qa_system
        
        return metadata


    async def get_answer(self, question: str, document_id: str) -> QuestionResponse:
        if document_id not in self.qa_systems:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        print(f"Question: {question}")
        qa_system = self.qa_systems[document_id]
        
        response = qa_system.query(question)
        
        return QuestionResponse(
            answer=response.answer,
            source_documents=[
                SourceDocument(content=doc.content, metadata=doc.metadata)
                for doc in response.source_documents
            ],
            metadata={
                "document_id": document_id,
                "filename": self.documents[document_id].filename
            }
        )

    def cleanup_document(self, document_id: str):
        """Remove document and its associated resources"""
        if document_id in self.documents:
            # Remove vector store
            vector_store_path = self._get_vector_store_path(document_id)
            if os.path.exists(vector_store_path):
                shutil.rmtree(vector_store_path)
            
            # Remove from dictionaries
            del self.documents[document_id]
            del self.qa_systems[document_id]
