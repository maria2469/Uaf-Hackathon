# summarizer_agent.py
from langchain_groq import ChatGroq
from typing import TypedDict, Dict, List
import json

class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: Dict
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]

def generate_patient_summary_func(patient_state: PatientState) -> str:
    """
    Generate a concise, actionable summary for a patient
    """
    llm = ChatGroq(temperature=0)

    labs_str = json.dumps(patient_state["labs"], indent=2)
    risks_str = ", ".join(patient_state["flagged_risks"]) if patient_state["flagged_risks"] else "No significant risks detected"
    actions_str = ", ".join(patient_state["actions_taken"]) if patient_state["actions_taken"] else "No actions taken yet"

    prompt = f"""
You are a medical summarization assistant.

Patient Name: {patient_state['patient_name']}
Labs: {labs_str}
Flagged Risks: {risks_str}
Actions Taken: {actions_str}

Task:
- Summarize this patient’s current health status in 2-3 concise sentences.
- Highlight the most urgent risks first.
- Include suggested follow-ups if actions have been taken.

Return only plain text summary.
"""

    try:
        result = llm.predict(prompt)

        # Handle AIMessage object
        if hasattr(result, "content"):  
            summary = result.content
        else:
            summary = str(result)

        return summary.strip()

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return f"Patient {patient_state['patient_name']} has the following risks: {risks_str}. Actions: {actions_str}."
