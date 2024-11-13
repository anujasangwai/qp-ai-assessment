from fastapi import Depends, HTTPException, status
from typing import Dict
from dotenv import load_dotenv

from service import QAService
from models import *

import time
import logging
#logging.basicConfig(level=logging.DEBUG)

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional

app = FastAPI(
    title="Document QA API",
    description="API for document upload and question answering",
    version="1.0.0"
)

qa_service = QAService()

# Configure logging (optional)
logging.basicConfig(level=logging.DEBUG)


@app.middleware("http")
async def log_client_details(request: Request, call_next):
    client_ip = request.client.host  # Get client IP address
    user_agent = request.headers.get("user-agent", "unknown")  # Get User-Agent header
    method = request.method  # HTTP method (GET, POST, etc.)
    path = request.url.path  # Request path

    # Log the request details
    logging.info(f"Client IP: {client_ip} | User Agent: {user_agent} | Method: {method} | Path: {path}")
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Log the response time
    logging.info(f"Completed in {process_time:.2f} seconds with status code {response.status_code}")
    return response

@app.get("/ws/socket.io/")
async def socket_io_not_supported():
    return {"message": "WebSocket (Socket.io) is not supported."}

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload a document for QA processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    try:
        content = await file.read()
        metadata = await qa_service.process_document(content, file.filename)
        
        return DocumentUploadResponse(
            document_id=metadata.document_id,
            filename=metadata.filename,
            message="Document processed successfully",
            status="success"
        )
    except Exception as e:
        print(e)
        import traceback
        print("Exception occurred:", traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/qa/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
):
    """Ask a question about a processed document"""
    try:
        return await qa_service.get_answer(
            question=request.question,
            document_id=request.document_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
):
    """Delete a document and its associated resources"""
    if document_id not in qa_service.documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    try:
        qa_service.cleanup_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """List all processed documents"""
    return list(qa_service.documents.values())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)