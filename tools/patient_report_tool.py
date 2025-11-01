"""
Patient Report Tool using LangChain Tool with Pydantic

This tool retrieves patient discharge reports.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from langchain.tools import tool
from utils.helpers import get_patient_by_name


class PatientReportInput(BaseModel):
    """Input schema for patient report tool."""
    patient_name: str = Field(description="Full name of the patient to retrieve report for")


@tool("fetch_patient_report", args_schema=PatientReportInput)
def fetch_patient_report(patient_name: str) -> str:
    """
    Retrieve patient discharge report by name.
    
    Args:
        patient_name: Full name of the patient to retrieve report for
        
    Returns:
        Patient discharge report as formatted string
    """
    patient = get_patient_by_name(patient_name)
    
    if not patient:
        return f"Patient '{patient_name}' not found in records"
    
    # Format patient report
    report = f"""Patient: {patient.get('patient_name', 'N/A')}
Discharge Date: {patient.get('discharge_date', 'N/A')}
Primary Diagnosis: {patient.get('primary_diagnosis', 'N/A')}

Medications:
{chr(10).join(f'  - {med}' for med in patient.get('medications', []))}

Dietary Restrictions: {patient.get('dietary_restrictions', 'N/A')}
Follow-up: {patient.get('follow_up', 'N/A')}
Warning Signs: {patient.get('warning_signs', 'N/A')}
Discharge Instructions: {patient.get('discharge_instructions', 'N/A')}"""
    return report


class PatientReportTool:
    """
    Tool for retrieving patient discharge reports.
    
    Uses Pydantic AI tools pattern for structured tool definition.
    """
    
    """
    Wrapper class for LangChain tool compatibility.
    Use fetch_patient_report directly as a tool.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize wrapper (tool is already registered)."""
        self.tool = fetch_patient_report
    
    def __call__(self, patient_name: str) -> Dict[str, Any]:
        """Call the tool and return dict result."""
        patient = get_patient_by_name(patient_name)
        if not patient:
            return {"error": f"Patient '{patient_name}' not found in records"}
        return patient

