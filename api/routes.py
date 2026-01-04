"""
API Routes for Post Discharge Medical AI Assistant

This module defines all API endpoints for the medical assistant.
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from agents.receptionist_agent import ReceptionistAgent
from agents.clinical_agent import ClinicalAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from rag.vectorstore import get_vectorstore
from rag.embeddings import get_embedding_model
from rag.loader import build_vectorstore_from_pdf
from utils.logger_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    patient_name: Optional[str] = None
    agent_type: Optional[str] = "receptionist"  # receptionist or clinical
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    agent_type: str
    conversation_id: str
    metadata: Optional[Dict[str, Any]] = None


class PatientReportRequest(BaseModel):
    """Request model for patient report retrieval."""
    patient_name: str


class InitializeRequest(BaseModel):
    """Request model for agent initialization endpoint."""
    force_rebuild_vectorstore: Optional[bool] = False


class InitializeResponse(BaseModel):
    """Response model for agent initialization endpoint."""
    success: bool
    message: str
    agents_initialized: List[str]
    timestamp: str


# Agents stored as module-level variables
receptionist_agent = None
clinical_agent = None
agents_initialized = False


def initialize_agents(force_rebuild_vectorstore: bool = False) -> Dict[str, Any]:
    """
    Initialize all agents for the multi-agent system.
    
    Args:
        force_rebuild_vectorstore: If True, force rebuild the vectorstore from PDF
        
    Returns:
        Dictionary with initialization status and agents
    """
    global receptionist_agent, clinical_agent, agents_initialized
    
    try:
        logger.info("Initializing agents and resources...")
        
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        get_embedding_model()
        
        # Check if vectorstore exists, if not build it from PDF
        logger.info("Checking vectorstore...")
        vectorstore = get_vectorstore()
        if not vectorstore or force_rebuild_vectorstore:
            logger.info("Vectorstore not found or rebuild requested. Building from PDF...")
            pdf_path = settings.NEPHROLOGY_PDF_PATH
            if os.path.exists(pdf_path):
                success = build_vectorstore_from_pdf(pdf_path, force_rebuild=force_rebuild_vectorstore)
                if success:
                    logger.info("Vectorstore built successfully")
                    vectorstore = get_vectorstore()
                else:
                    logger.warning("Failed to build vectorstore. RAG may not work properly.")
            else:
                logger.warning(f"PDF file not found: {pdf_path}. RAG may not work properly.")
        
        # Initialize LLM
        logger.info("Initializing LLM (Gemini)...")
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found. LLM functionality may be limited.")
            raise ValueError("GEMINI_API_KEY not found. Cannot initialize agents.")
        
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Initialize clinical agent first (needed for receptionist routing)
        logger.info("Initializing clinical agent...")
        clinical_agent = ClinicalAgent(llm, vectorstore)
        
        # Initialize receptionist agent with reference to clinical agent
        logger.info("Initializing receptionist agent...")
        receptionist_agent = ReceptionistAgent(llm, clinical_agent=clinical_agent)
        
        agents_initialized = True
        logger.info("All agents initialized successfully")
        
        return {
            "success": True,
            "receptionist": receptionist_agent,
            "clinical": clinical_agent
        }
    except Exception as e:
        logger.error(f"Error initializing agents: {e}", exc_info=True)
        agents_initialized = False
        raise


@router.post("/initialize", response_model=InitializeResponse)
async def initialize_agents_endpoint(request: InitializeRequest = InitializeRequest()):
    """
    Initialize agents for the medical assistant system.
    
    This endpoint must be called before using the /chat endpoint.
    
    Args:
        request: Initialization request with optional force_rebuild_vectorstore flag
        
    Returns:
        Initialization response with status and initialized agents
    """
    try:
        result = initialize_agents(force_rebuild_vectorstore=request.force_rebuild_vectorstore)
        
        agents_list = []
        if receptionist_agent:
            agents_list.append("receptionist")
        if clinical_agent:
            agents_list.append("clinical")
        
        return InitializeResponse(
            success=True,
            message="Agents initialized successfully",
            agents_initialized=agents_list,
            timestamp=datetime.now().isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize agents: {str(e)}")


@router.get("/initialize/status")
async def get_initialization_status():
    """
    Check if agents are initialized.
    
    Returns:
        Status of agent initialization
    """
    return {
        "initialized": agents_initialized,
        "receptionist_agent": receptionist_agent is not None,
        "clinical_agent": clinical_agent is not None,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for interacting with the medical assistant.
    
    Args:
        request: Chat request with message and optional patient name
        
    Returns:
        Chat response from the appropriate agent
    """
    # Check if agents are initialized
    if not agents_initialized or not receptionist_agent or not clinical_agent:
        raise HTTPException(
            status_code=503,
            detail="Agents not initialized. Please call /api/v1/initialize first."
        )
    
    # Get the appropriate agent
    agent = receptionist_agent
    if request.agent_type == "clinical":
        agent = clinical_agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not available")
    
    # Process request through agent
    try:
        response = agent.invoke(
            message=request.message,
            patient_name=request.patient_name,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=response.get("response", "No response"),
            agent_type=response.get("agent_type", request.agent_type),
            conversation_id=request.conversation_id or "default",
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/patient-report")
async def get_patient_report(request: PatientReportRequest):
    """
    Retrieve patient discharge report.
    
    Args:
        request: Patient name to retrieve report for
        
    Returns:
        Patient discharge report
    """
    from tools.patient_report_tool import fetch_patient_report
    
    # Use patient report tool
    result = fetch_patient_report(request.patient_name)
    
    if "not found" in result or "not available" in result:
        raise HTTPException(status_code=404, detail=result)
    
    return {"patient_name": request.patient_name, "report": result}


@router.get("/agents")
async def list_agents():
    """
    List available agents and their capabilities.
    
    Returns:
        List of available agents
    """
    # TODO: Return agent information
    return {
        "agents": [
            {
                "name": "receptionist",
                "description": "Handles initial queries and routing",
                "capabilities": ["patient_lookup", "appointment_scheduling"]
            },
            {
                "name": "clinical",
                "description": "Handles clinical questions and medical advice",
                "capabilities": ["rag_retrieval", "web_search", "symptom_analysis"]
            }
        ]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history.
    
    Args:
        conversation_id: Unique conversation identifier
        
    Returns:
        Conversation history
    """
    # Placeholder - conversation storage to be implemented
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "metadata": {}
    }

