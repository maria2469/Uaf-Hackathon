from pydantic import BaseModel
from typing import List, Dict, Optional

class LabFlags(BaseModel):
    HighRiskDiseases: Optional[List[str]] = []
    AbnormalBloodPressure: Optional[List[str]] = []
    AbnormalCholesterol: Optional[List[str]] = []
    PositiveFindings: Optional[List[str]] = []

class LabResult(BaseModel):
    patient_name: str
    flags: LabFlags
    explanation: str

class PatientSummary(BaseModel):
    patient_id: str
    name: str
    summary: str
    referral_draft: Optional[str] = ""
    raw_model_output: Optional[str] = ""
    explanation: Optional[str] = ""

class ReferralResult(BaseModel):
    status: str
    specialist: Optional[str] = None
    reason: Optional[str] = None
    doctor: Optional[str] = None
