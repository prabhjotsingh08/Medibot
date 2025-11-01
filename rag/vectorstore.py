"""
Vectorstore Implementation using ChromaDB

This module handles ChromaDB vector database operations for RAG.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from rag.embeddings import get_embedding_model
from config.settings import settings

# Initialize ChromaDB client and vectorstore
_vectorstore = None
_chroma_client = None


def get_chroma_client():
    """
    Get or create ChromaDB client instance.
    
    Returns:
        ChromaDB client
    """
    global _chroma_client
    
    if _chroma_client is None:
        persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
    
    return _chroma_client


def get_vectorstore(collection_name: Optional[str] = None) -> Optional[Chroma]:
    """
    Get or create LangChain Chroma vectorstore instance.
    
    Args:
        collection_name: Optional collection name
        
    Returns:
        LangChain Chroma vectorstore or None if not initialized
    """
    global _vectorstore
    
    if _vectorstore is None:
        collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        embedding_model = get_embedding_model()
        persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        try:
            # Try to load existing vectorstore
            _vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=persist_directory
            )
        except Exception:
            # If no existing vectorstore, create empty one
            _vectorstore = None
    
    return _vectorstore


def initialize_vectorstore(documents: List[str], metadatas: List[Dict[str, Any]], 
                          ids: Optional[List[str]] = None) -> Chroma:
    """
    Initialize vectorstore with documents.
    
    Args:
        documents: List of document texts
        metadatas: List of metadata dictionaries
        ids: Optional list of document IDs
        
    Returns:
        Initialized Chroma vectorstore
    """
    embedding_model = get_embedding_model()
    collection_name = settings.CHROMA_COLLECTION_NAME
    persist_directory = settings.CHROMA_PERSIST_DIRECTORY
    
    # Create Chroma instance with documents
    vectorstore = Chroma.from_texts(
        texts=documents,
        metadatas=metadatas,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
        ids=ids
    )
    
    global _vectorstore
    _vectorstore = vectorstore
    
    return vectorstore


