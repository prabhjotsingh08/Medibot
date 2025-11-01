"""
RAG Retrieval Tool for Clinical Agent

This tool retrieves medical knowledge from ChromaDB vectorstore.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain.tools import tool
from rag.vectorstore import get_vectorstore
from config.settings import settings
from utils.logger_config import get_logger
import logfire
import os
from datetime import datetime

logger = get_logger(__name__)


class RAGRetrievalInput(BaseModel):
    """Input schema for RAG retrieval tool."""
    query: str = Field(description="Medical query to search in knowledge base")
    k: Optional[int] = Field(default=5, description="Number of results to retrieve")


@tool("rag_retrieval", args_schema=RAGRetrievalInput)
def rag_retrieval(query: str, k: int = 5) -> str:
    """
    Retrieve relevant medical knowledge from ChromaDB vectorstore.
    
    Args:
        query: Medical query to search
        k: Number of results to retrieve
        
    Returns:
        Retrieved medical knowledge as formatted string with citations
    """
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] rag_retrieval called with query: {query[:50]}...")
        vectorstore = get_vectorstore()
        
        if not vectorstore:
            logger.warning("Vectorstore not initialized")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [WARNING] Vectorstore not initialized")
            return "Medical knowledge base not available. Please initialize the vectorstore first."
        
        # Search vectorstore
        docs = vectorstore.similarity_search_with_score(query, k=k)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] rag_retrieval completed: {len(docs)} results")
        
        if not docs:
            return f"No relevant information found for query: {query}"
        
        # Format results with citations
        formatted_results = f"Retrieved medical information for '{query}':\n\n"
        for i, (doc, score) in enumerate(docs, 1):
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            
            formatted_results += f"--- Reference {i} (Relevance: {score:.3f}) ---\n"
            import os
            formatted_results += f"Source: {os.path.basename(source)}, Page {page}\n"
            formatted_results += f"Content: {doc.page_content}\n\n"
        
        return formatted_results.strip()
        
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] RAG retrieval error: {e}")
        return f"Error retrieving medical knowledge: {str(e)}"


# Helper function for programmatic access
def retrieve_medical_knowledge(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve medical knowledge as structured data.
    
    Args:
        query: Medical query
        k: Number of results
        
    Returns:
        List of retrieved documents with metadata
    """
    try:
        import os
        vectorstore = get_vectorstore()
        
        if not vectorstore:
            return []
        
        docs = vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs:
            metadata = doc.metadata
            results.append({
                "text": doc.page_content,
                "metadata": metadata,
                "score": float(score),
                "citation": f"{os.path.basename(metadata.get('source', 'Unknown'))}, Page {metadata.get('page', 'N/A')}"
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving medical knowledge: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] Error retrieving medical knowledge: {e}")
        return []

