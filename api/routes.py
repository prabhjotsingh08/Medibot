"""
API Routes for Post Discharge Medical AI Assistant

This module defines all API endpoints for the medical assistant.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from agents.receptionist_agent import ReceptionistAgent
from agents.clinical_agent import ClinicalAgent

router = APIRouter()


# TODO: Define Pydantic models for request/response schemas
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


# Agents will be initialized in server.py and stored in app.state
receptionist_agent = None
clinical_agent = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for interacting with the medical assistant.
    
    Args:
        request: Chat request with message and optional patient name
        
    Returns:
        Chat response from the appropriate agent
    """
    # Get agents from module-level variables (set in main.py)
    # This will be properly wired in server startup
    agent = receptionist_agent
    if request.agent_type == "clinical":
        agent = clinical_agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agents not initialized")
    
    # Process request through agent
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

