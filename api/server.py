"""
FastAPI Server for Post Discharge Medical AI Assistant

This module contains the main FastAPI application setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config.settings import settings

# Initialize FastAPI app with proper metadata
app = FastAPI(
    title="Post Discharge Medical AI Assistant",
    description="Multi-agent AI system for post-discharge medical assistance",
    version="0.1.0"
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Configure CORS origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["medical-assistant"])


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "healthy", "service": "Post Discharge Medical AI Assistant"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Note: Could add actual health checks (database, vectorstore, etc.) in future
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # Initialize vectorstore and agents will be set from main.py
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    # Cleanup if needed
    pass

