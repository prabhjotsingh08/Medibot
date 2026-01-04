"""
Document Chunking Implementation

This module handles document chunking strategies for RAG with improved text cleaning.
"""

import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config.settings import settings


def clean_text(text: str) -> str:
    """
    Clean text extracted from PDF to improve quality.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Replace special unicode characters with standard equivalents
    replacements = {
        '\ue01f': 'fi',  # fi ligature
        '\ue01e': 'fl',  # fl ligature
        '\ue020': ' ',   # special space
        '\ue021': '!',
        '\xa0': ' ',     # non-breaking space
        '\u200b': '',    # zero-width space
        '\ufeff': '',    # byte order mark
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Fix broken words at line breaks (but preserve intentional line breaks)
    text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation at line breaks
    
    # Normalize multiple newlines to double newlines (paragraph breaks)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Replace single newlines with spaces (for flowing text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Clean up any remaining multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_text_splitter(chunk_size: Optional[int] = None, 
                     chunk_overlap: Optional[int] = None) -> RecursiveCharacterTextSplitter:
    """
    Get recursive character text splitter instance.
    
    Args:
        chunk_size: Size of each chunk (defaults to settings.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to settings.CHUNK_OVERLAP)
        
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True
    )


def chunk_text(text: str, chunk_size: Optional[int] = None, 
              chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Chunk a text string into smaller pieces.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (defaults to settings.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to settings.CHUNK_OVERLAP)
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Clean the text first
    text = clean_text(text)
    
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunk_stripped = chunk.strip()
        
        # Skip very short chunks (likely noise)
        if len(chunk_stripped) < 50:
            continue
            
        chunk_dict = {
            "text": chunk_stripped,
            "metadata": {
                "source": "text_input",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk_stripped)
            }
        }
        chunked_docs.append(chunk_dict)
    
    return chunked_docs


def chunk_documents(documents: List[Any], 
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents (supports LangChain Document objects or dicts).
    
    Args:
        documents: List of LangChain Document objects or dicts with 'page_content'/'text' and 'metadata'
        chunk_size: Size of each chunk (defaults to settings.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to settings.CHUNK_OVERLAP)
        
    Returns:
        List of chunked documents with metadata
    """
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        # Handle both LangChain Document objects and dictionaries
        if hasattr(doc, 'page_content'):
            text = doc.page_content
            metadata = doc.metadata.copy() if doc.metadata else {}
        else:
            text = doc.get("text", doc.get("page_content", ""))
            metadata = doc.get("metadata", {}).copy()
        
        # Clean the text
        text = clean_text(text)
        
        # Skip very short pages (likely blank or noise)
        if len(text.strip()) < 50:
            continue
        
        # Split into chunks
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_stripped = chunk.strip()
            
            # Skip very short chunks
            if len(chunk_stripped) < 50:
                continue
            
            # Create chunk metadata (preserve original metadata and add chunk info)
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk_stripped),
                "original_page": metadata.get("page", 0)
            }
            
            chunk_dict = {
                "text": chunk_stripped,
                "metadata": chunk_metadata
            }
            all_chunks.append(chunk_dict)
    
    return all_chunks


def chunk_pdf(pdf_path: str, chunk_size: Optional[int] = None,
             chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and chunk a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Size of each chunk (defaults to settings.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to settings.CHUNK_OVERLAP)
        
    Returns:
        List of chunked documents with metadata
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = get_text_splitter(chunk_size, chunk_overlap)
        chunks = []
        
        for page in pages:
            # Clean page content
            page_text = clean_text(page.page_content)
            
            # Skip very short pages
            if len(page_text.strip()) < 50:
                continue
            
            page_chunks = text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(page_chunks):
                chunk_stripped = chunk.strip()
                
                # Skip very short chunks
                if len(chunk_stripped) < 50:
                    continue
                    
                chunk_dict = {
                    "text": chunk_stripped,
                    "metadata": {
                        "source": pdf_path,
                        "page": page.metadata.get("page", 0),
                        "chunk_index": i,
                        "total_pages": len(pages),
                        "char_count": len(chunk_stripped)
                    }
                }
                chunks.append(chunk_dict)
        
        return chunks
        
    except Exception as e:
        print(f"Error chunking PDF {pdf_path}: {e}")
        return []