"""Paper loading and document processing utilities."""

import asyncio
import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime

import aiofiles
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import structlog

from ..config import settings
from ..utils.memory import cache_manager

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """Base class for document processors."""

    def __init__(self):
        self.supported_types = []

    async def can_process(self, file_path: str, mime_type: str) -> bool:
        """Check if this processor can handle the file type."""
        return mime_type in self.supported_types

    async def process(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process the document and return Document objects."""
        raise NotImplementedError


class PDFProcessor(DocumentProcessor):
    """PDF document processor using pypdf."""

    def __init__(self):
        super().__init__()
        self.supported_types = ["application/pdf"]

    async def process(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process PDF file and extract text."""
        try:
            documents = []

            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()

                        if text.strip():  # Only add pages with content
                            doc_metadata = {
                                **metadata,
                                "page": page_num + 1,
                                "total_pages": len(pdf_reader.pages),
                                "source_type": "pdf"
                            }

                            document = Document(
                                page_content=text,
                                metadata=doc_metadata
                            )
                            documents.append(document)

                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}", error=str(e))
                        continue

            logger.info(f"PDF processed successfully",
                       file=file_path,
                       pages=len(documents))

            return documents

        except Exception as e:
            logger.error(f"Error processing PDF", file=file_path, error=str(e))
            raise


class TextProcessor(DocumentProcessor):
    """Plain text document processor."""

    def __init__(self):
        super().__init__()
        self.supported_types = ["text/plain", "text/markdown"]

    async def process(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process text file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()

            doc_metadata = {
                **metadata,
                "source_type": "text"
            }

            document = Document(
                page_content=content,
                metadata=doc_metadata
            )

            logger.info(f"Text file processed successfully", file=file_path)
            return [document]

        except Exception as e:
            logger.error(f"Error processing text file", file=file_path, error=str(e))
            raise


class PaperLoader:
    """
    Production-grade paper loading and processing system.

    Features:
    - Multiple file format support
    - Async processing
    - Text chunking and optimization
    - Metadata extraction
    - Error handling and validation
    - Caching for performance
    """

    def __init__(self):
        self.processors = [
            PDFProcessor(),
            TextProcessor(),
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)

    async def validate_file(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """Validate uploaded file."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }

        # Check file size
        if file_size > settings.max_file_size:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"File size ({file_size} bytes) exceeds maximum allowed size ({settings.max_file_size} bytes)"
            )

        # Check file type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            validation_result["warnings"].append("Could not determine file type")
            mime_type = "application/octet-stream"

        # Check if file type is supported
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        if file_extension not in settings.allowed_file_types:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"File type '{file_extension}' is not supported. "
                f"Allowed types: {', '.join(settings.allowed_file_types)}"
            )

        # Check if we have a processor for this file type
        can_process = False
        for processor in self.processors:
            if await processor.can_process(file_path, mime_type):
                can_process = True
                break

        if not can_process:
            validation_result["valid"] = False
            validation_result["errors"].append(f"No processor available for file type: {mime_type}")

        validation_result["file_info"] = {
            "mime_type": mime_type,
            "file_extension": file_extension,
            "file_size": file_size
        }

        return validation_result

    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk."""
        # Generate unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file_hash}_{filename}"

        file_path = os.path.join(settings.upload_dir, safe_filename)

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        logger.info(f"File saved", original_name=filename, saved_path=file_path)
        return file_path

    async def extract_metadata(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from file."""
        file_stat = os.stat(file_path)

        metadata = {
            "filename": filename,
            "file_path": file_path,
            "file_size": file_stat.st_size,
            "upload_time": datetime.now().isoformat(),
            "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        }

        # Try to extract additional metadata based on file type
        mime_type, _ = mimetypes.guess_type(file_path)
        metadata["mime_type"] = mime_type

        # For PDFs, try to extract title and author
        if mime_type == "application/pdf":
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    if pdf_reader.metadata:
                        if pdf_reader.metadata.title:
                            metadata["title"] = pdf_reader.metadata.title
                        if pdf_reader.metadata.author:
                            metadata["author"] = pdf_reader.metadata.author
                        if pdf_reader.metadata.subject:
                            metadata["subject"] = pdf_reader.metadata.subject
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata", error=str(e))

        return metadata

    async def process_document(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process document using appropriate processor."""
        mime_type = metadata.get("mime_type")

        # Find appropriate processor
        processor = None
        for proc in self.processors:
            if await proc.can_process(file_path, mime_type):
                processor = proc
                break

        if not processor:
            raise ValueError(f"No processor found for file type: {mime_type}")

        # Process the document
        documents = await processor.process(file_path, metadata)

        # Split documents into chunks if they're too large
        chunked_documents = []
        for doc in documents:
            if len(doc.page_content) > settings.chunk_size:
                chunks = self.text_splitter.split_documents([doc])
                chunked_documents.extend(chunks)
            else:
                chunked_documents.append(doc)

        logger.info(f"Document processed and chunked",
                   original_docs=len(documents),
                   chunked_docs=len(chunked_documents))

        return chunked_documents

    async def load_from_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Load and process a document from uploaded file content.

        Args:
            file_content: Raw file content as bytes
            filename: Original filename

        Returns:
            Processing result with documents and metadata
        """
        result = {
            "success": False,
            "documents": [],
            "metadata": {},
            "errors": [],
            "warnings": []
        }

        try:
            # Save uploaded file
            file_path = await self.save_uploaded_file(file_content, filename)

            # Validate file
            validation = await self.validate_file(file_path, len(file_content))
            result["warnings"].extend(validation["warnings"])

            if not validation["valid"]:
                result["errors"].extend(validation["errors"])
                return result

            # Extract metadata
            metadata = await self.extract_metadata(file_path, filename)
            result["metadata"] = metadata

            # Process document
            documents = await self.process_document(file_path, metadata)
            result["documents"] = documents
            result["success"] = True

            logger.info(f"Document loaded successfully",
                       filename=filename,
                       document_count=len(documents))

        except Exception as e:
            logger.error(f"Error loading document", filename=filename, error=str(e))
            result["errors"].append(f"Processing error: {str(e)}")

        return result

    async def load_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        Load and process a document from file path.

        Args:
            file_path: Path to the file

        Returns:
            Processing result with documents and metadata
        """
        result = {
            "success": False,
            "documents": [],
            "metadata": {},
            "errors": [],
            "warnings": []
        }

        try:
            if not os.path.exists(file_path):
                result["errors"].append(f"File not found: {file_path}")
                return result

            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # Validate file
            validation = await self.validate_file(file_path, file_size)
            result["warnings"].extend(validation["warnings"])

            if not validation["valid"]:
                result["errors"].extend(validation["errors"])
                return result

            # Extract metadata
            metadata = await self.extract_metadata(file_path, filename)
            result["metadata"] = metadata

            # Process document
            documents = await self.process_document(file_path, metadata)
            result["documents"] = documents
            result["success"] = True

            logger.info(f"Document loaded from path",
                       file_path=file_path,
                       document_count=len(documents))

        except Exception as e:
            logger.error(f"Error loading document from path", file_path=file_path, error=str(e))
            result["errors"].append(f"Processing error: {str(e)}")

        return result

    async def batch_load(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Load multiple documents in batch.

        Args:
            file_paths: List of file paths to process

        Returns:
            Batch processing results
        """
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "documents": [],
            "errors": [],
            "file_results": []
        }

        for file_path in file_paths:
            try:
                file_result = await self.load_from_path(file_path)
                results["file_results"].append({
                    "file_path": file_path,
                    "success": file_result["success"],
                    "document_count": len(file_result["documents"]),
                    "errors": file_result["errors"]
                })

                if file_result["success"]:
                    results["successful"] += 1
                    results["documents"].extend(file_result["documents"])
                else:
                    results["failed"] += 1
                    results["errors"].extend(file_result["errors"])

            except Exception as e:
                results["failed"] += 1
                error_msg = f"Error processing {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        logger.info(f"Batch processing completed",
                   total=results["total_files"],
                   successful=results["successful"],
                   failed=results["failed"])

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats."""
        formats = {}
        for processor in self.processors:
            processor_name = processor.__class__.__name__
            formats[processor_name] = processor.supported_types

        return formats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the paper loader."""
        status = {
            "processors_available": len(self.processors),
            "supported_formats": self.get_supported_formats(),
            "upload_dir_exists": os.path.exists(settings.upload_dir),
            "upload_dir_writable": os.access(settings.upload_dir, os.W_OK),
            "text_splitter_configured": self.text_splitter is not None
        }

        # Test text splitting
        try:
            test_text = "This is a test document. " * 100
            chunks = self.text_splitter.split_text(test_text)
            status["text_splitting_test"] = len(chunks) > 1
        except Exception as e:
            status["text_splitting_test"] = False
            status["text_splitting_error"] = str(e)

        return status


# Factory function
def create_paper_loader() -> PaperLoader:
    """Factory function to create a paper loader instance."""
    return PaperLoader()

