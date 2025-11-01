"""
API Data Models for Post Discharge Medical AI Assistant

This module contains Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# TODO: Add more comprehensive request/response models
class BaseRequest(BaseModel):
    """Base request model."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class PatientInfo(BaseModel):
    """Patient information model."""
    patient_name: str
    discharge_date: str
    primary_diagnosis: str
    medications: List[str]
    dietary_restrictions: str
    follow_up: str
    warning_signs: str
    discharge_instructions: str


class AgentCapability(BaseModel):
    """Agent capability model."""
    name: str
    description: str
    capabilities: List[str]


# TODO: Add more models as needed:
# - Conversation model
# - Message model
# - Error response model
# - Analytics model

