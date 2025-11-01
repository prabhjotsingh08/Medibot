"""
Document Chunking Implementation

This module handles document chunking strategies for RAG.
"""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from config.settings import settings


def get_text_splitter(chunk_size: Optional[int] = None, 
                     chunk_overlap: Optional[int] = None,
                     strategy: str = "recursive") -> RecursiveCharacterTextSplitter:
    """
    Get text splitter instance.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy ("recursive" or "character")
        
    Returns:
        Text splitter instance
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    if strategy == "recursive":
        # TODO: Configure recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        # TODO: Configure character text splitter
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    return text_splitter


def chunk_text(text: str, chunk_size: Optional[int] = None, 
              chunk_overlap: Optional[int] = None, 
              add_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    Chunk a text string into smaller pieces.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        add_metadata: Whether to add metadata to chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    # TODO: Implement text chunking
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    # TODO: Format chunks with metadata
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        if add_metadata:
            chunk_dict["metadata"] = {
                "source": "text_input",
                "chunk_index": i
            }
        chunked_docs.append(chunk_dict)
    
    return chunked_docs


def chunk_documents(documents: List[Dict[str, Any]], 
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of document dictionaries with 'text' and 'metadata'
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents with metadata
    """
    # TODO: Implement document chunking
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        # TODO: Split document text
        chunks = text_splitter.split_text(text)
        
        # TODO: Add chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            all_chunks.append(chunk_dict)
    
    return all_chunks


def chunk_pdf(pdf_path: str, chunk_size: Optional[int] = None,
             chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and chunk a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents with metadata
    """
    # TODO: Implement PDF chunking
    # TODO: Load PDF using PyPDFLoader
    # TODO: Extract text and metadata (page numbers, etc.)
    # TODO: Chunk the extracted text
    # TODO: Preserve page number information in metadata
    
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = get_text_splitter(chunk_size, chunk_overlap)
        chunks = []
        
        for page in pages:
            page_chunks = text_splitter.split_text(page.page_content)
            for i, chunk in enumerate(page_chunks):
                chunk_dict = {
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": page.metadata.get("page", 0),
                        "chunk_index": i
                    }
                }
                chunks.append(chunk_dict)
        
        return chunks
    except Exception as e:
        # TODO: Handle errors gracefully
        print(f"Error chunking PDF {pdf_path}: {e}")
        return []

