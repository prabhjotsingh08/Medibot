"""
Settings Configuration

This module handles all configuration settings for the application.
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE: float = os.getenv("LLM_TEMPERATURE", 0.0)
    # Embedding Configuration (use HuggingFace by default - free)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Web Search Configuration
    USE_WEB_SEARCH: bool = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    WEB_SEARCH_API: str = os.getenv("WEB_SEARCH_API", "serpapi")  # serpapi only
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "medical_knowledge")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Data Paths
    PATIENT_REPORTS_PATH: str = os.getenv("PATIENT_REPORTS_PATH", "./data/patient_reports.json")
    NEPHROLOGY_PDF_PATH: str = os.getenv("NEPHROLOGY_PDF_PATH", "./data/nephrology/comprehensive-clinical-nephrology.pdf")
    
    # FastAPI Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    CORS_ORIGINS: List[str] = ["*"]  # Configure properly for production
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables (e.g., LOGFIRE_TOKEN)


# Singleton instance
settings = Settings()

