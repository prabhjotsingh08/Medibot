"""
Helper Utility Functions

This module contains utility functions used across the application.
"""

import json
import os
from typing import Dict, Any, Optional, List


def load_patient_data(data_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load patient reports from JSON file.
    
    Args:
        data_path: Path to patient reports JSON file
        
    Returns:
        List of patient records
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "patient_reports.json"
        )
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []


def get_patient_by_name(patient_name: str, data_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get patient record by name (case-insensitive).
    
    Args:
        patient_name: Patient name to search for
        data_path: Optional path to patient reports JSON file
        
    Returns:
        Patient record dictionary or None if not found
    """
    patient_data = load_patient_data(data_path)
    if not patient_data:
        return None
    
    patient_name_lower = patient_name.lower().strip()
    for patient in patient_data:
        if patient.get("patient_name", "").lower() == patient_name_lower:
            return patient
    
    return None


def format_patient_report(patient: Dict[str, Any]) -> str:
    """
    Format patient report as readable string.
    
    Args:
        patient: Patient record dictionary
        
    Returns:
        Formatted patient report string
    """
    formatted = f"""
Patient Name: {patient.get('patient_name', 'N/A')}
Discharge Date: {patient.get('discharge_date', 'N/A')}
Primary Diagnosis: {patient.get('primary_diagnosis', 'N/A')}

Medications:
{chr(10).join(f'  - {med}' for med in patient.get('medications', []))}

Dietary Restrictions: {patient.get('dietary_restrictions', 'N/A')}

Follow-up: {patient.get('follow_up', 'N/A')}

Warning Signs: {patient.get('warning_signs', 'N/A')}

Discharge Instructions: {patient.get('discharge_instructions', 'N/A')}
"""
    return formatted.strip()


def extract_patient_name(text: str) -> Optional[str]:
    """
    Extract patient name from text using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        Extracted patient name or None
    """
    import re
    
    # Common patterns for names
    patterns = [
        r"my name is (\w+\s+\w+)",
        r"i'm (\w+\s+\w+)",
        r"i am (\w+\s+\w+)",
        r"name is (\w+\s+\w+)",
        r"patient name[:\s]+(\w+\s+\w+)",
        r"patient\s+name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            name = match.group(1).strip().title()
            return name
    
    # Check for capitalized words (potential names)
    words = text.split()
    if len(words) >= 2:
        # Look for two consecutive capitalized words
        for i in range(len(words) - 1):
            if words[i][0].isupper() and words[i+1][0].isupper() and words[i].isalpha() and words[i+1].isalpha():
                name = f"{words[i]} {words[i+1]}"
                return name
    
    return None


