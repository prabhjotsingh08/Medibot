"""
Embeddings Implementation

This module handles embedding model setup and configuration.
Uses all-MiniLM-L6-v2 as default (ChromaDB default, free, local).
"""

from typing import List
from langchain_core.embeddings import Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)

# Initialize embedding model
_embedding_model = None


def get_embedding_model() -> Embeddings:
    """
    Get or create embedding model instance.
    Uses all-MiniLM-L6-v2 by default (free, local, ChromaDB default).
    
    Returns:
        Embedding model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        embedding_provider = settings.EMBEDDING_PROVIDER.lower()
        
        if embedding_provider == "huggingface" or embedding_provider == "local":
            # Use all-MiniLM-L6-v2 (default, free, local)
            logger.info("Initializing HuggingFace embeddings (all-MiniLM-L6-v2)")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing HuggingFace embeddings (all-MiniLM-L6-v2)")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            # Default to all-MiniLM-L6-v2
            logger.info("Using default HuggingFace embeddings (all-MiniLM-L6-v2)")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Using default HuggingFace embeddings (all-MiniLM-L6-v2)")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    return _embedding_model


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)


def embed_documents(documents: List[str]) -> List[List[float]]:
    """
    Embed multiple documents.
    
    Args:
        documents: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_documents(documents)

