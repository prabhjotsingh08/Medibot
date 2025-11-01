"""
PDF Loader and Embedding Builder

This module handles loading PDFs, chunking, and building embeddings for ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vectorstore import initialize_vectorstore
from rag.chunking import get_text_splitter
from rag.embeddings import get_embedding_model
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)


def load_and_chunk_pdf(pdf_path: str, chunk_size: Optional[int] = None, 
                      chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load PDF and chunk it into smaller pieces.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents with metadata
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] PDF file not found: {pdf_path}")
        return []
    
    logger.info(f"Loading PDF: {pdf_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Loading PDF: {pdf_path}")
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        logger.info(f"Loaded {len(pages)} pages from PDF")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Loaded {len(pages)} pages from PDF")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Chunk documents
        chunks = []
        for page in pages:
            page_chunks = text_splitter.split_text(page.page_content)
            for i, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": pdf_path,
                        "page": page.metadata.get("page", 0),
                        "chunk_index": i,
                        "total_pages": len(pages)
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks from PDF")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Created {len(chunks)} chunks from PDF")
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error loading PDF {pdf_path}: {e}")
        return []


def build_vectorstore_from_pdf(pdf_path: Optional[str] = None, 
                              collection_name: Optional[str] = None,
                              force_rebuild: bool = False) -> bool:
    """
    Build ChromaDB vectorstore from PDF.
    
    Args:
        pdf_path: Path to PDF file (defaults to nephrology PDF)
        collection_name: ChromaDB collection name
        force_rebuild: Force rebuild even if vectorstore exists
        
    Returns:
        True if successful
    """
    pdf_path = pdf_path or settings.NEPHROLOGY_PDF_PATH
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] PDF file not found: {pdf_path}")
        return False
    
    # Check if vectorstore already exists
    from rag.vectorstore import get_vectorstore
    existing_vectorstore = get_vectorstore(collection_name)
    
    if existing_vectorstore and not force_rebuild:
        logger.info("Vectorstore already exists. Use force_rebuild=True to rebuild.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Vectorstore already exists. Use force_rebuild=True to rebuild.")
        return True
    
    # Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        logger.error("No chunks created from PDF")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] No chunks created from PDF")
        return False
    
    # Extract texts and metadatas
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Generate IDs
    ids = [f"chunk_{i}" for i in range(len(texts))]
    
    # Initialize vectorstore
    logger.info(f"Building vectorstore with {len(texts)} chunks...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Building vectorstore with {len(texts)} chunks...")
    try:
        vectorstore = initialize_vectorstore(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        if vectorstore:
            logger.info("Vectorstore built successfully")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Vectorstore built successfully")
            return True
        else:
            logger.error("Failed to build vectorstore")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Failed to build vectorstore")
            return False
            
    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error building vectorstore: {e}")
        return False


if __name__ == "__main__":
    # Set up logging
    from utils.logger_config import setup_logging
    setup_logging()
    
    # Build vectorstore from nephrology PDF
    success = build_vectorstore_from_pdf(force_rebuild=False)
    if success:
        print("Vectorstore built successfully!")
    else:
        print("Failed to build vectorstore")

