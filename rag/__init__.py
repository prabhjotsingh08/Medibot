"""
RAG Module for Post Discharge Medical AI Assistant

This module handles vectorstore, embeddings, and chunking for RAG implementation.
"""

from rag.vectorstore import get_vectorstore, initialize_vectorstore
from rag.embeddings import get_embedding_model
from rag.chunking import chunk_documents, chunk_pdf
from rag.loader import build_vectorstore_from_pdf, load_and_chunk_pdf

__all__ = [
    "get_vectorstore",
    "initialize_vectorstore",
    "get_embedding_model",
    "chunk_documents",
    "chunk_pdf",
    "build_vectorstore_from_pdf",
    "load_and_chunk_pdf"
]

