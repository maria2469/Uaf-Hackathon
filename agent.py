import time
import json
import sqlite3
from typing import List, TypedDict, Dict, Set
from summarizer_agent import generate_patient_summary_func
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# ====== Agent State ======
class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: List[Dict]  # Store history of lab reports
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]

# ====== Database Layer ======
DB_PATH = "healthcare.db"
MEMORY_PATH = "patient_memory.json"

def fetch_all_lab_reports(batch_size: int = 50, offset: int = 0) -> List[Dict]:
    """Fetch a batch of lab reports"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lab_reports LIMIT ? OFFSET ?", (batch_size, offset))
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    data = [dict(zip(cols, row)) for row in rows]
    conn.close()
    return data

def fetch_all_doctors() -> List[Dict]:
    """Fetch all doctors once"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    doctors = [dict(zip(cols, row)) for row in rows]
    conn.close()
    return doctors

# ====== LLM Safe Call ======
def safe_llm_invoke(prompt: str, llm_model) -> Dict:
    """Invoke LLM safely, parse JSON, fallback if invalid"""
    try:
        response = llm_model.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        if not content:
            return {}
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract JSON from text
        start, end = content.find("{"), content.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end])
            except:
                return {}
        return {}
    except Exception as e:
        print(f"❌ LLM invocation error: {e}")
        return {}

# ====== LLM-Based Risk & Doctor Reasoning ======
def llm_assess_patients(patients: List[Dict], doctors: List[Dict], llm_model) -> Dict[str, Dict]:
    """Send batch of patients + doctors to LLM and get structured risks + doctor recommendations"""
    if not patients:
        return {}
    prompt = f"""
You are a medical reasoning assistant.
Given the following patients and doctors, for each patient:
1. Identify potential health risks from labs/symptoms.
2. Suggest the most appropriate doctors (by name) for each risk if necessary.
3. Return ONLY JSON in the format:

{{
    "UID_00001": {{
        "risks": ["risk1", "risk2"],
        "recommended_doctors": ["Dr. X", "Dr. Y"]
    }},
    ...
}}

Patients:
{json.dumps(patients)}

Doctors:
{json.dumps(doctors)}
"""
    return safe_llm_invoke(prompt, llm_model)

# ====== Summarization Agent ======
def generate_summary(patient_state: PatientState) -> str:
    try:
        return generate_patient_summary_func(patient_state)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return "Summary unavailable."

# ====== Load/Save Memory ======
def load_memory(path: str) -> Dict[str, PatientState]:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {pid: PatientState(**state) for pid, state in data.items()}
    return {}

def save_memory(path: str, memory: Dict[str, PatientState]):
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)

# ====== Main Agent Loop ======
if __name__ == "__main__":
    llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    doctors = fetch_all_doctors()

    # Load persistent memory
    patient_memory: Dict[str, PatientState] = load_memory(MEMORY_PATH)
    processed_patients: Set[str] = set(patient_memory.keys())

    offset = 0
    batch_size = 50

    try:
        while True:
            # Fetch batch of patients
            patients_batch = fetch_all_lab_reports(batch_size=batch_size, offset=offset)
            
            # Detect new patients
            new_patients = [p for p in patients_batch if p.get("unique_id") not in processed_patients]

            if not new_patients:
                print("✅ No new patients in this batch. Waiting for new data...")
                offset = 0
                time.sleep(30)
                continue

            # LLM assessment
            assessments = llm_assess_patients(new_patients, doctors, llm_model)

            for lab_data in new_patients:
                patient_id = lab_data.get("unique_id")
                patient_name = lab_data.get("name", "Unknown")
                assessment = assessments.get(patient_id, {})

                flagged_risks = assessment.get("risks", [])
                recommended_doctors = assessment.get("recommended_doctors", [])

                # Merge with memory if patient exists
                if patient_id in patient_memory:
                    prev_state = patient_memory[patient_id]
                    # Append new labs
                    labs_history = prev_state["labs"] + [lab_data]
                    flagged_risks = list(set(prev_state["flagged_risks"] + flagged_risks))
                    actions_taken = prev_state["actions_taken"]
                else:
                    labs_history = [lab_data]
                    actions_taken = []

                patient_state: PatientState = {
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "labs": labs_history,
                    "flagged_risks": flagged_risks,
                    "summary": "",
                    "actions_taken": actions_taken
                }

                # Generate summary
                patient_state["summary"] = generate_summary(patient_state)

                # Append doctor actions
                for doc_name in recommended_doctors:
                    action_msg = f"Recommended doctor: {doc_name}"
                    if action_msg not in patient_state["actions_taken"]:
                        patient_state["actions_taken"].append(action_msg)

                # Update memory
                patient_memory[patient_id] = patient_state
                processed_patients.add(patient_id)

                # Display patient dashboard
                print(f"\nPatient: {patient_state['patient_name']}")
                print(f"Flagged Risks: {patient_state['flagged_risks']}")
                print(f"Summary: {patient_state['summary']}")
                print(f"Actions Taken: {patient_state['actions_taken']}")

            # Save memory after each batch
            save_memory(MEMORY_PATH, patient_memory)

            offset += batch_size
            time.sleep(5)

    except KeyboardInterrupt:
        print("🛑 UPI Agent stopped by user.")
        save_memory(MEMORY_PATH, patient_memory)
