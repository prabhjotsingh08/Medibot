"""
PDF Loader and Embedding Builder

This module handles loading PDFs, chunking, and building embeddings for ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from rag.vectorstore import initialize_vectorstore, delete_collection
from rag.chunking import chunk_documents
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)


def load_and_chunk_pdf(pdf_path: str, chunk_size: Optional[int] = None, 
                      chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load PDF and chunk it into smaller pieces using recursive character-based chunking.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Size of each chunk (defaults to settings.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to settings.CHUNK_OVERLAP)
        
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
        
        # Chunk documents using the centralized chunking function
        chunks = chunk_documents(pages, chunk_size, chunk_overlap)
        
        logger.info(f"Created {len(chunks)} chunks from PDF")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Created {len(chunks)} chunks from PDF")
        
        # Log sample of first chunk for verification
        if chunks:
            sample_text = chunks[0]['text'][:200]
            logger.info(f"Sample chunk (first 200 chars): {sample_text}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Sample chunk: {sample_text}...")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error loading PDF: {e}")
        return []


def build_vectorstore_from_pdf(pdf_path: Optional[str] = None, 
                              collection_name: Optional[str] = None,
                              force_rebuild: bool = False) -> bool:
    """
    Build ChromaDB vectorstore from PDF.
    Always deletes existing collection when force_rebuild=True.
    
    Args:
        pdf_path: Path to PDF file (defaults to nephrology PDF from settings)
        collection_name: ChromaDB collection name (defaults to settings)
        force_rebuild: If True, deletes existing collection and rebuilds from scratch
        
    Returns:
        True if successful, False otherwise
    """
    pdf_path = pdf_path or settings.NEPHROLOGY_PDF_PATH
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] PDF file not found: {pdf_path}")
        return False
    
    # Always delete existing collection when force_rebuild is True
    if force_rebuild:
        logger.info("Force rebuild enabled - deleting existing collection...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Deleting existing collection...")
        delete_collection(collection_name)
    
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
            ids=ids,
            collection_name=collection_name
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
        logger.error(f"Error building vectorstore: {e}", exc_info=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error building vectorstore: {e}")
        return False


if __name__ == "__main__":
    # Set up logging
    from utils.logger_config import setup_logging
    setup_logging()
    
    # Build vectorstore from nephrology PDF with force rebuild
    print("Building vectorstore with force rebuild...")
    success = build_vectorstore_from_pdf(force_rebuild=True)
    if success:
        print("✅ Vectorstore built successfully!")
    else:
        print("❌ Failed to build vectorstore")