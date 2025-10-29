from langchain_groq import ChatGroq
from typing import TypedDict, Dict, List
import json

class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: List[Dict]             # Updated to match your ingestion loop
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]

def generate_patient_summary_func(patient_state: PatientState, llm: ChatGroq = None) -> str:
    """
    Generates a structured clinical summary for a patient.
    Accepts a preloaded LLM instance for efficiency.
    """
    if llm is None:
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    # Concatenate all lab entries for the summary
    labs_str = ""
    for lab in patient_state.get("labs", []):
        labs_str += json.dumps(lab, indent=2) + "\n"

    risks_str = ", ".join(patient_state.get("flagged_risks", [])) or "No significant risks detected"
    actions_str = ", ".join(patient_state.get("actions_taken", [])) or "No actions taken yet"

    prompt = f"""
You are a medical summarization assistant. Generate a structured clinical summary **using ONLY the provided data**.

Patient Name: {patient_state['patient_name']}
Labs: {labs_str.strip()}
Flagged Risks: {risks_str}
Actions Taken: {actions_str}

Instructions:
1. Patient Overview
   - Include age/gender only if available in labs
   - 2-3 concise sentences
2. Urgent Risks
   - List severe vs moderate separately if possible
3. Lab Highlights
   - Only abnormal/noteworthy labs
   - Latest lab values first
4. Actions Taken / Recommendations
   - Include actions already performed
   - Suggest realistic next steps
5. Follow-Up
   - Labs, specialist referral, monitoring

Formatting Rules:
- Use headings: Patient Overview, Urgent Risks, Lab Highlights, Actions Taken / Recommendations, Follow-Up
- Bullet points for clarity
- Return output in plain text only
- Do NOT hallucinate data; use only labs, risks, actions provided
"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        # Fallback: plain text summary
        return (
            f"Patient {patient_state['patient_name']}\n"
            f"Flagged Risks: {risks_str}\n"
            f"Actions Taken: {actions_str}\n"
            f"Labs Summary:\n{labs_str.strip()[:500]}..."  # first 500 chars
        )
