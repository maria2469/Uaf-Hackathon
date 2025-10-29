from pathlib import Path
from dotenv import load_dotenv  # ✅ Add this
load_dotenv() 
from src.Agents.orchestrator import Coordinator   # ✅ correct path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "mock_data" / "healthcare.db"

coordinator = Coordinator(db_path=DB_PATH)

def run_patient_workflow(patient_id: str):
    result = coordinator.run_patient_workflow(patient_id)
    return {
        "patient_id": patient_id,
        "lab_flags": result.get("lab_flags"),
        "lab_explanation": result.get("lab_explanation"),
        "summary": result.get("summary"),
        "referral_draft": result.get("referral_draft"),
    }
