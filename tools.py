"""Tools for the Patient Monitoring ReAct Agent.

Includes functions for:
- Fetching patient lab reports
- Retrieving doctor info
- Updating memory with feedback
- Generating patient summaries
"""

from typing import Any, Callable, Dict, List
import sqlite3
from langgraph.runtime import get_runtime
from context import Context
from summarizer_agent import generate_patient_summary_func
import json

from state import State

# ===== Database helpers =====
DB_PATH = "healthcare.db"

def fetch_lab_reports(patient_name: str) -> List[Dict[str, Any]]:
    """Fetch all lab reports for a given patient."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lab_reports WHERE name = ?", (patient_name,))
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

def fetch_doctors() -> List[Dict[str, Any]]:
    """Fetch all doctors from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

# ====== ReAct Tools =====

async def get_patient_labs(patient_name: str, state: State) -> List[Dict[str, Any]]:
    """Tool to retrieve patient lab reports for the agent."""
    return fetch_lab_reports(patient_name)

async def get_doctor_list(_: Any, state: State) -> List[Dict[str, Any]]:
    """Tool to retrieve doctor info."""
    return fetch_doctors()

async def update_patient_feedback(feedback_text: str, state: State) -> str:
    """Tool to store clinician feedback in the agent's memory snapshot."""
    if not state.patient_name:
        return "No patient selected for feedback."

    memory = state.patient_memory_snapshot
    patient_name = state.patient_name

    # Ensure patient record exists
    for patient_id, patient in memory.items():
        if patient["patient_name"].lower() == patient_name.lower():
            if "feedback" not in patient:
                patient["feedback"] = []
            patient["feedback"].append(feedback_text)
            return f"Feedback added for patient '{patient_name}'."

    return f"No records found for patient '{patient_name}'."

async def generate_summary_tool(_: Any, state: State) -> str:
    """Tool to generate a summary for the currently selected patient."""
    patient_name = state.patient_name
    if not patient_name:
        return "No patient selected for summary."

    # Get patient memory snapshot
    for patient in state.patient_memory_snapshot.values():
        if patient["patient_name"].lower() == patient_name.lower():
            try:
                summary = generate_patient_summary_func(patient)
                return summary
            except Exception as e:
                return f"Error generating summary: {e}"

    return f"No records found for patient '{patient_name}'."

# ====== Export tools list =====
TOOLS: List[Callable[..., Any]] = [
    get_patient_labs,
    get_doctor_list,
    update_patient_feedback,
    generate_summary_tool,
]
