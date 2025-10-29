import time
import json
import sqlite3
from typing import List, TypedDict, Dict, Set
from summarizer_agent import generate_patient_summary_func
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import threading
import re

load_dotenv()

# ====== Agent State ======
class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: List[Dict]
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]

# ====== Database Layer ======
DB_PATH = "healthcare.db"
MEMORY_PATH = "patient_memory.json"

def fetch_all_lab_reports(batch_size: int = 50, offset: int = 0) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lab_reports LIMIT ? OFFSET ?", (batch_size, offset))
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

def fetch_all_doctors() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

# ====== LLM Safe Call ======
def safe_llm_invoke(prompt: str, llm_model) -> str:
    try:
        response = llm_model.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        print(f"❌ LLM invocation error: {e}")
        return "LLM unavailable"

# ====== LLM Patient Assessment ======
def llm_assess_patients(patients: List[Dict], doctors: List[Dict], llm_model) -> Dict[str, Dict]:
    if not patients:
        return {}
    prompt = f"""
You are a medical reasoning assistant.
Given the following patients and doctors, for each patient:
1. Identify potential health risks from labs/symptoms.
2. Suggest the most appropriate doctors (by name) for each risk if necessary.
3. Return ONLY JSON in the format:

{{
    "UID_00001": {{"risks": ["risk1", "risk2"], "recommended_doctors": ["Dr. X", "Dr. Y"]}}
}}

Patients:
{json.dumps(patients)}

Doctors:
{json.dumps(doctors)}
"""
    content = safe_llm_invoke(prompt, llm_model)
    try:
        return json.loads(content)
    except:
        return {}

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
        json.dump(memory, f, indent=2, sort_keys=True)

# ====== Extract Patient Name from Query ======
def extract_patient_name(query: str) -> str:
    query = query.strip()
    
    # Common patterns for patient queries
    patterns = [
        r"medical history of ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"patient report ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"how is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"condition of ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*) doing",  
        r"is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*) ok",      
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.I)
        if match:
            return match.group(1).strip()
    
    # fallback: look for any sequence of capitalized words
    match = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", query)
    return match.group(1).strip() if match else ""

# ====== Query-Time LLM Reasoning ======
def llm_query_patient(name: str, memory: Dict[str, PatientState], llm_model, full_report: bool = False) -> str:
    for patient in memory.values():
        if patient["patient_name"].lower() == name.lower():
            if full_report:
                # Detailed report
                prompt = f"""
You are a clinical assistant. Provide a FULL medical report for {name}.
Patient record:
{json.dumps(patient, indent=2)}

Include:
- Flagged risks
- Lab highlights
- Suggested next steps
- Urgent attention needed

Respond in readable plain text.
"""
            else:
                # Concise update
                prompt = f"""
You are a clinical assistant. A doctor asks: "How is {name} doing?"
Patient record:
{json.dumps(patient, indent=2)}

Provide a concise, 2-3 sentence update:
- Current flagged risks
- Lab highlights
- Urgent concerns or next steps
"""
            return safe_llm_invoke(prompt, llm_model)
    return f"No records found for patient {name}"

# ====== Background Ingestion Loop ======
def ingestion_loop(patient_memory: Dict[str, PatientState], processed_patients: Set[str], llm_model, doctors):
    offset = 0
    batch_size = 50
    while True:
        patients_batch = fetch_all_lab_reports(batch_size=batch_size, offset=offset)
        new_patients = [p for p in patients_batch if p.get("unique_id") not in processed_patients]

        if new_patients:
            assessments = llm_assess_patients(new_patients, doctors, llm_model)
            for lab_data in new_patients:
                patient_id = lab_data.get("unique_id")
                patient_name = lab_data.get("name", "Unknown")
                assessment = assessments.get(patient_id, {})

                flagged_risks = assessment.get("risks", [])
                recommended_doctors = assessment.get("recommended_doctors", [])

                if patient_id in patient_memory:
                    prev_state = patient_memory[patient_id]
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

                patient_state["summary"] = generate_summary(patient_state)

                for doc_name in recommended_doctors:
                    action_msg = f"Recommended doctor: {doc_name}"
                    if action_msg not in patient_state["actions_taken"]:
                        patient_state["actions_taken"].append(action_msg)

                patient_memory[patient_id] = patient_state
                processed_patients.add(patient_id)

            save_memory(MEMORY_PATH, patient_memory)
            offset += batch_size
        else:
            offset = 0
        time.sleep(5)

# ====== Main Console Loop ======
if __name__ == "__main__":
    llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    doctors = fetch_all_doctors()
    patient_memory: Dict[str, PatientState] = load_memory(MEMORY_PATH)
    processed_patients: Set[str] = set(patient_memory.keys())

    # Start background ingestion
    ingestion_thread = threading.Thread(
        target=ingestion_loop,
        args=(patient_memory, processed_patients, llm_model, doctors),
        daemon=True
    )
    ingestion_thread.start()

    try:
        while True:
            query = input("\nType a query (or 'exit' to quit): ").strip()
            if query.lower() == "exit":
                break

            name = extract_patient_name(query)
            if not name:
                print("❌ Could not extract patient name from query.")
                continue

            # Determine query type
            if "medical history of" in query.lower() or "patient report" in query.lower():
                print(llm_query_patient(name, patient_memory, llm_model, full_report=True))
            elif "how is" in query.lower() or "condition of" in query.lower() or "is" in query.lower():
                print(llm_query_patient(name, patient_memory, llm_model, full_report=False))
            else:
                # Generic fallback
                print(f"✅ Noted your query: '{query}'. No patient report generated.")

    except KeyboardInterrupt:
        print("\n🛑 UPI Agent stopped by user.")
    finally:
        save_memory(MEMORY_PATH, patient_memory)
