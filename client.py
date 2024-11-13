import httpx
import os
from typing import List, Optional
from models import DocumentUploadResponse, QuestionResponse, DocumentMetadata, QuestionRequest
from pydantic import BaseModel

# Define the base URL for the API
BASE_URL = "http://127.0.0.1:8000"

import logging

# logging.basicConfig(level=logging.DEBUG)

class DocumentQAClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def upload_document(self, file_path: str) -> DocumentUploadResponse:
        """Uploads a document to the server."""
        with open(file_path, "rb") as file:
            file_data = {"file": (file_path, file, "application/pdf")}
            print(file_path)
            response = await self.client.post("/documents/upload", files=file_data, timeout=30.0)
            response.raise_for_status()
            return DocumentUploadResponse(**response.json())

    async def ask_question(self, document_id: str, question: str) -> QuestionResponse:
        """Asks a question about an uploaded document."""
        question_data = QuestionRequest(question=question, document_id=document_id)
        response = await self.client.post("/qa/question", json=question_data.dict(), timeout=30.0)
        response.raise_for_status()
        return QuestionResponse(**response.json())

    async def delete_document(self, document_id: str) -> dict:
        """Deletes a document and its resources."""
        response = await self.client.delete(f"/documents/{document_id}")
        response.raise_for_status()
        return response.json()

    async def list_documents(self) -> List[DocumentMetadata]:
        """Lists all processed documents."""
        response = await self.client.get("/documents")
        response.raise_for_status()
        return [DocumentMetadata(**doc) for doc in response.json()]

    async def close(self):
        """Closes the client session."""
        await self.client.aclose()


async def main():
    client = DocumentQAClient()

    try:
        file_name = os.getcwd() + "/source_data.pdf"
        upload_response = await client.upload_document(file_name)
        print("Uploaded document:", upload_response)
    except httpx.HTTPStatusError as e:
        print("Error uploading document:", e)

    # Ask a question
    try:
        while True:
            query = input("Enter you query: ")
            if "exit" in query:
                break
            question_response = await client.ask_question(upload_response.document_id, query)
            print("Question response:", question_response.answer)
            print(f"sources: {question_response.source_documents}")
    except httpx.HTTPStatusError as e:
        print("Error asking question:", e)

    # List all documents
    try:
        documents = await client.list_documents()
        print("All documents:", documents)
    except httpx.HTTPStatusError as e:
        print("Error listing documents:", e)

    # Delete the uploaded document
    try:
        delete_response = await client.delete_document(upload_response.document_id)
        print("Deleted document:", delete_response)
    except httpx.HTTPStatusError as e:
        print("Error deleting document:", e)

    # Close the client session
    await client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
