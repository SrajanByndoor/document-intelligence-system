#!/usr/bin/env python3
"""
Document Intelligence System - FastAPI REST API

This module provides a RESTful API interface for the Document Intelligence System,
enabling PDF document upload, question answering, and summarization through HTTP endpoints.

Endpoints:
    POST /upload - Upload a PDF document
    POST /query - Ask a question about an uploaded document
    POST /summarize - Generate a summary of an uploaded document
    GET /health - Health check endpoint
    DELETE /document/{file_id} - Delete an uploaded document

Usage:
    uvicorn api:app --reload
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import main.py functions
import main

# =============================================================================
# CONFIGURATION
# =============================================================================

UPLOAD_DIR = Path("./uploads")
ALLOWED_EXTENSIONS = {".pdf"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for document query endpoint.

    Attributes:
        file_id: Unique identifier of the uploaded document.
        question: The question to ask about the document.

    Example:
        {
            "file_id": "abc123-def456",
            "question": "What was the total revenue?"
        }
    """
    file_id: str = Field(..., description="Unique identifier of the uploaded document")
    question: str = Field(..., min_length=1, description="Question to ask about the document")


class SummarizeRequest(BaseModel):
    """Request model for document summarization endpoint.

    Attributes:
        file_id: Unique identifier of the uploaded document.
        max_length: Maximum length of the summary in words (optional).

    Example:
        {
            "file_id": "abc123-def456",
            "max_length": 150
        }
    """
    file_id: str = Field(..., description="Unique identifier of the uploaded document")
    max_length: Optional[int] = Field(
        default=150,
        ge=50,
        le=500,
        description="Maximum summary length in words"
    )


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    filename: str
    message: str
    file_id: str


class SummaryResponse(BaseModel):
    """Response model for summarization endpoint."""
    summary: str
    original_length: int
    summary_length: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    timestamp: str


class DeleteResponse(BaseModel):
    """Response model for document deletion endpoint."""
    message: str
    deleted: bool


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Document Intelligence System API",
    description="REST API for PDF document analysis, question answering, and summarization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    return response


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize application on startup.

    - Creates upload directory if it doesn't exist
    - Pre-loads ML models to avoid cold start on first request
    """
    logger.info("Starting Document Intelligence System API...")

    # Create uploads directory
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory ready: {UPLOAD_DIR.absolute()}")

    # Pre-load models by importing modules (they load on import)
    logger.info("Pre-loading ML models...")
    try:
        # These imports trigger model loading
        import text_qa
        import table_qa
        import summarizer
        import question_router
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")
        raise

    logger.info("API startup complete")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_file_path(file_id: str) -> Path:
    """
    Get the file path for a given file ID.

    Args:
        file_id: Unique identifier of the file.

    Returns:
        Path to the file in the uploads directory.
    """
    return UPLOAD_DIR / f"{file_id}.pdf"


def validate_file_exists(file_id: str) -> Path:
    """
    Validate that a file exists and return its path.

    Args:
        file_id: Unique identifier of the file.

    Returns:
        Path to the file.

    Raises:
        HTTPException: If file is not found (404).
    """
    file_path = get_file_path(file_id)
    if not file_path.exists():
        logger.warning(f"File not found: {file_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with file_id '{file_id}' not found"
        )
    return file_path


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for analysis.

    Accepts a PDF file upload, validates the file type, generates a unique ID,
    and saves the file to the uploads directory.

    **Args:**
        - file: PDF file to upload (multipart/form-data)

    **Returns:**
        - filename: Original name of the uploaded file
        - message: Success message
        - file_id: Unique identifier for the uploaded document

    **Raises:**
        - 400: If file is not a PDF
        - 500: If file save fails

    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/upload" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@document.pdf"
    ```

    **Example Response:**
    ```json
    {
        "filename": "document.pdf",
        "message": "File uploaded successfully",
        "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
    ```
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file type attempted: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only PDF files are allowed. Got: {file_ext}"
        )

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = get_file_path(file_id)

    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"File uploaded: {file.filename} -> {file_id}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    return UploadResponse(
        filename=file.filename,
        message="File uploaded successfully",
        file_id=file_id
    )


@app.post(
    "/query",
    summary="Query a document",
    tags=["Analysis"]
)
async def query_document(request: QueryRequest):
    """
    Ask a question about an uploaded document.

    Uses the document intelligence pipeline to answer questions about the content
    of an uploaded PDF document. The system automatically routes the question to
    the appropriate QA model (text or table-based).

    **Args:**
        - file_id: Unique identifier of the uploaded document
        - question: Natural language question about the document

    **Returns:**
        Full answer dictionary from the main pipeline including:
        - question: The original question
        - answer: The extracted answer
        - score: Confidence score (0.0 to 1.0)
        - source_type: 'text' or 'table'
        - source_page: Page number where answer was found
        - routing_decision: How the question was routed
        - success: Whether an answer was found
        - message: Status message
        - processing_time: Time taken to process

    **Raises:**
        - 400: If question is empty
        - 404: If document not found
        - 500: If processing fails

    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/query" \
         -H "Content-Type: application/json" \
         -d '{"file_id": "abc123", "question": "What was the total revenue?"}'
    ```

    **Example Response:**
    ```json
    {
        "question": "What was the total revenue?",
        "answer": "$1.2 billion",
        "score": 0.89,
        "source_type": "table",
        "source_page": 5,
        "routing_decision": "table",
        "success": true,
        "message": "Answer found in table",
        "processing_time": 1.234
    }
    ```
    """
    # Validate file exists
    file_path = validate_file_exists(request.file_id)

    # Validate question
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    # Process query
    try:
        logger.info(f"Processing query for {request.file_id}: {request.question[:50]}...")
        result = main.process_document_query(
            pdf_path=str(file_path),
            question=request.question
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Summarize a document",
    tags=["Analysis"]
)
async def summarize_document(request: SummarizeRequest):
    """
    Generate a summary of an uploaded document.

    Uses the BART summarization model to generate a concise summary of the
    document content.

    **Args:**
        - file_id: Unique identifier of the uploaded document
        - max_length: Maximum summary length in words (default: 150, range: 50-500)

    **Returns:**
        - summary: The generated summary text
        - original_length: Word count of original document
        - summary_length: Word count of the generated summary

    **Raises:**
        - 404: If document not found
        - 500: If summarization fails

    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/summarize" \
         -H "Content-Type: application/json" \
         -d '{"file_id": "abc123", "max_length": 100}'
    ```

    **Example Response:**
    ```json
    {
        "summary": "This document discusses the company's annual performance...",
        "original_length": 5000,
        "summary_length": 95
    }
    ```
    """
    # Validate file exists
    file_path = validate_file_exists(request.file_id)

    # Process summarization
    try:
        logger.info(f"Summarizing document: {request.file_id}")
        result = main.summarize_document_content(
            pdf_path=str(file_path),
            max_length=request.max_length,
            min_length=request.max_length // 3  # Set min to ~1/3 of max
        )

        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('message', 'Summarization failed')
            )

        return SummaryResponse(
            summary=result['summary'],
            original_length=result['original_length'],
            summary_length=result['summary_length']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize document: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"]
)
async def health_check():
    """
    Simple health check endpoint.

    Returns the current health status of the API and a timestamp.

    **Returns:**
        - status: Health status ('healthy')
        - timestamp: Current server time in ISO format

    **Example Request:**
    ```bash
    curl -X GET "http://localhost:8000/health"
    ```

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00.000000"
    }
    ```
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.delete(
    "/document/{file_id}",
    response_model=DeleteResponse,
    summary="Delete a document",
    tags=["Documents"]
)
async def delete_document(file_id: str):
    """
    Delete an uploaded document.

    Removes the document file from the server permanently.

    **Args:**
        - file_id: Unique identifier of the document to delete (path parameter)

    **Returns:**
        - message: Status message
        - deleted: Boolean indicating if deletion was successful

    **Raises:**
        - 404: If document not found
        - 500: If deletion fails

    **Example Request:**
    ```bash
    curl -X DELETE "http://localhost:8000/document/abc123-def456"
    ```

    **Example Response:**
    ```json
    {
        "message": "Document deleted successfully",
        "deleted": true
    }
    ```
    """
    # Validate file exists
    file_path = validate_file_exists(file_id)

    # Delete file
    try:
        os.remove(file_path)
        logger.info(f"Document deleted: {file_id}")
        return DeleteResponse(
            message="Document deleted successfully",
            deleted=True
        )
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run the API server directly using:
        python api.py

    Or with uvicorn for development:
        uvicorn api:app --reload

    For production:
        uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

    ============================================================================
    EXAMPLE CURL COMMANDS FOR TESTING
    ============================================================================

    1. Health Check:
        curl -X GET "http://localhost:8000/health"

    2. Upload a PDF:
        curl -X POST "http://localhost:8000/upload" \
             -H "Content-Type: multipart/form-data" \
             -F "file=@document.pdf"

    3. Query a Document (replace FILE_ID with actual ID from upload):
        curl -X POST "http://localhost:8000/query" \
             -H "Content-Type: application/json" \
             -d '{"file_id": "FILE_ID", "question": "What was the total revenue?"}'

    4. Summarize a Document:
        curl -X POST "http://localhost:8000/summarize" \
             -H "Content-Type: application/json" \
             -d '{"file_id": "FILE_ID", "max_length": 150}'

    5. Delete a Document:
        curl -X DELETE "http://localhost:8000/document/FILE_ID"

    ============================================================================
    INTERACTIVE API DOCUMENTATION
    ============================================================================

    Once the server is running, visit:
        - Swagger UI: http://localhost:8000/docs
        - ReDoc: http://localhost:8000/redoc
    """
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
