"""
Vectorstore Implementation using ChromaDB

This module handles ChromaDB vector database operations for RAG.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import shutil
import os
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from rag.embeddings import get_embedding_model
from config.settings import settings
from utils.logger_config import get_logger
from datetime import datetime

logger = get_logger(__name__)

# Global state for vectorstore (singleton pattern)
_vectorstore = None
_chroma_client = None


def delete_collection(collection_name: Optional[str] = None) -> bool:
    """
    Delete the ChromaDB collection and reset global state.
    
    Args:
        collection_name: Name of collection to delete (defaults to settings.CHROMA_COLLECTION_NAME)
        
    Returns:
        True if successful, False otherwise
    """
    global _vectorstore, _chroma_client
    
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    persist_directory = settings.CHROMA_PERSIST_DIRECTORY
    
    logger.info(f"Deleting collection '{collection_name}'...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Deleting collection '{collection_name}'...")
    
    try:
        # Reset global variables
        _vectorstore = None
        _chroma_client = None
        
        # Delete the entire persist directory
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            logger.info(f"Deleted vectorstore directory: {persist_directory}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Deleted vectorstore directory: {persist_directory}")
        
        # Recreate empty directory
        os.makedirs(persist_directory, exist_ok=True)
        
        logger.info(f"Collection '{collection_name}' deleted successfully")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Collection deleted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error deleting collection: {e}")
        return False


def get_chroma_client():
    """
    Get or create ChromaDB client instance (singleton).
    
    Returns:
        ChromaDB PersistentClient
    """
    global _chroma_client
    
    if _chroma_client is None:
        persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    return _chroma_client


def get_vectorstore(collection_name: Optional[str] = None) -> Optional[Chroma]:
    """
    Get existing LangChain Chroma vectorstore instance.
    Does NOT create a new one if it doesn't exist.
    
    Args:
        collection_name: Optional collection name (defaults to settings.CHROMA_COLLECTION_NAME)
        
    Returns:
        LangChain Chroma vectorstore or None if not initialized
    """
    global _vectorstore
    
    # Return cached instance if available
    if _vectorstore is not None:
        return _vectorstore
    
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    embedding_model = get_embedding_model()
    persist_directory = settings.CHROMA_PERSIST_DIRECTORY
    
    # Check if persist directory exists and has data
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.info("No existing vectorstore found")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No existing vectorstore found")
        return None
    
    try:
        logger.info("Loading existing vectorstore...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Loading existing vectorstore...")
        
        # Load existing vectorstore
        _vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        
        # Verify it has documents
        try:
            count = _vectorstore._collection.count()
            if count == 0:
                logger.warning("Vectorstore exists but is empty")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] Vectorstore is empty")
                _vectorstore = None
            else:
                logger.info(f"Loaded existing vectorstore with {count} documents")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Loaded vectorstore with {count} documents")
        except Exception as e:
            logger.warning(f"Could not verify document count: {e}")
            
    except Exception as e:
        logger.info(f"Could not load existing vectorstore: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Could not load existing vectorstore: {e}")
        _vectorstore = None
    
    return _vectorstore


def initialize_vectorstore(documents: List[str], metadatas: List[Dict[str, Any]], 
                          ids: Optional[List[str]] = None,
                          collection_name: Optional[str] = None) -> Chroma:
    """
    Initialize a NEW vectorstore with documents.
    This should only be called after deleting any existing collection.
    
    Args:
        documents: List of document texts
        metadatas: List of metadata dictionaries
        ids: Optional list of document IDs
        collection_name: Optional collection name (defaults to settings.CHROMA_COLLECTION_NAME)
        
    Returns:
        Initialized Chroma vectorstore
    """
    global _vectorstore
    
    embedding_model = get_embedding_model()
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    persist_directory = settings.CHROMA_PERSIST_DIRECTORY
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    logger.info(f"Creating new vectorstore with {len(documents)} documents...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Creating vectorstore with {len(documents)} documents...")
    
    try:
        # Create Chroma instance with documents
        _vectorstore = Chroma.from_texts(
            texts=documents,
            metadatas=metadatas,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            ids=ids
        )
        
        logger.info("Vectorstore created and cached successfully")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Vectorstore created successfully")
        
        return _vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}", exc_info=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error creating vectorstore: {e}")
        _vectorstore = None
        raise