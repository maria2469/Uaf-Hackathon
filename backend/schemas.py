# models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class LabReport(BaseModel):
    id: int
    unique_id: str
    name: str
    report_data: Dict

class PatientState(BaseModel):
    patient_id: str
    patient_name: str
    labs: List[Dict]
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]
    feedback: List[str]
