import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from config import QAConfig, Document, DocumentChunk

class DocumentProcessor:
    def __init__(self, config: QAConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def load_documents(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return [Document(str(doc.page_content), doc.metadata) for doc in documents]

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        chunks = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc.content)
            for i, split in enumerate(splits):
                metadata = {
                    **doc.metadata,
                    "chunk_id": i,
                    "source_doc_id": len(chunks)
                }
                chunks.append(DocumentChunk(split, metadata))
        return chunks

