import time
import json
import sqlite3
import os
import threading
from typing import Dict, List, Set, TypedDict, Optional
from dotenv import load_dotenv
from difflib import get_close_matches
from threading import Lock
from langchain_groq import ChatGroq

# === Local modules ===
from agent.summarizer_agent import generate_patient_summary_func
from agent.referral_email import fetch_all_doctors, find_relevant_doctor, send_referral_email
from agent.utils import safe_llm_invoke

load_dotenv()

# =================== CONFIG ===================
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mock_data/healthcare.db"))

MEMORY_PATH = "patient_memory.json"
MEMORY_LOCK = Lock()

# =================== STATE TYPE ===================
class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: List[Dict]
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]
    feedback: List[str]

# =================== DB HELPERS ===================
def fetch_all_lab_reports(batch_size=50, offset=0) -> List[Dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM lab_reports LIMIT ? OFFSET ?", (batch_size, offset))
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        print(f"❌ DB error fetching labs: {e}")
        return []
    finally:
        conn.close()

# =================== MEMORY HELPERS ===================
def load_memory(path: str) -> Dict[str, PatientState]:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {pid: PatientState(**state) for pid, state in data.items()}
    return {}

def save_memory(path: str, memory: Dict[str, PatientState]):
    with MEMORY_LOCK:
        with open(path, "w") as f:
            json.dump(memory, f, indent=2, sort_keys=True)

# =================== SUMMARY ===================
def generate_summary(patient_state: PatientState) -> str:
    try:
        return generate_patient_summary_func(patient_state)
    except Exception as e:
        print(f"❌ Summary error: {e}")
        return "Summary unavailable."

# =================== FEEDBACK ===================
def submit_feedback(patient_name: str, feedback_text: str, memory: Dict[str, PatientState]):
    if not patient_name:
        print("❌ No patient specified for feedback.")
        return
    all_names = [p["patient_name"] for p in memory.values()]
    match = get_close_matches(patient_name, all_names, n=1, cutoff=0.6)
    if not match:
        print(f"❌ No patient found matching {patient_name}")
        return
    for pid, patient in memory.items():
        if patient["patient_name"] == match[0]:
            with MEMORY_LOCK:
                patient.setdefault("feedback", []).append(feedback_text)
                patient["summary"] = generate_summary(patient)
            print(f"✅ Feedback stored for {match[0]}")
            return

# =================== LLM INTERPRETATION ===================
def interpret_query(query: str, memory: Dict[str, PatientState], llm_model) -> Dict:
    all_patients = [p["patient_name"] for p in memory.values()]
    patient_list_str = ", ".join(all_patients) if all_patients else "No known patients."
    prompt = f"""
You are MEDI-AI, a unified autonomous hospital intelligence system.

Understand and classify the following user query:
"{query}"

Known patients: {patient_list_str}

Return STRICT JSON:
{{
  "intent": string,  # ["get_status","get_full_report","get_lab_results","get_vitals","get_critical_patients","submit_feedback","send_referral","other"]
  "patient_name": string or null,
  "feedback_text": string or null,
  "reasoning": string,
  "action_text": string
}}
"""
    try:
        raw = safe_llm_invoke(prompt, llm_model)
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        print(f"❌ interpret_query error: {e}")
        return {
            "intent": "other",
            "patient_name": None,
            "feedback_text": None,
            "reasoning": str(e),
            "action_text": "Noted the query."
        }

# =================== PATIENT QUERY VIA LLM ===================
AGENT_PROMPT = """
You are a clinical assistant AI. Always respond based on DB and memory data.
Include flagged risks, recent labs, and reasoning.
Never hallucinate new info.
"""

def llm_query_patient(name: str, memory: Dict[str, PatientState], llm_model, full_report=False) -> str:
    patient = next((p for p in memory.values() if p["patient_name"].lower() == name.lower()), None)
    if not patient:
        return f"No records found for patient {name}"
    patient_copy = dict(patient)
    patient_copy["labs"] = [patient["labs"][-1]] if patient["labs"] else []
    record = json.dumps(patient_copy, indent=2)
    instructions = "Provide a brief 2-line clinical summary." if not full_report else \
                   "Provide full structured report including reasoning and interpretation."
    prompt = f"{AGENT_PROMPT}\n\nPatient: {name}\nInstructions: {instructions}\nRecord:\n{record}"
    try:
        response = safe_llm_invoke(prompt, llm_model)
        return response or "No response from model."
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return "LLM query failed."

# =================== CRITICAL PATIENTS ===================
def get_critical_patients(memory: Dict[str, PatientState]) -> str:
    critical = [
        {"name": p["patient_name"], "id": p["patient_id"], "risk": ", ".join(p["flagged_risks"])}
        for p in memory.values() if p["flagged_risks"]
    ]
    return "No critical patients." if not critical else json.dumps(critical, indent=2)

# =================== UNIFIED QUERY HANDLER ===================
def handle_query(query: str, memory: Dict[str, PatientState], llm_model, doctors: List[Dict]):
    parsed = interpret_query(query, memory, llm_model)
    intent = parsed.get("intent")
    name = parsed.get("patient_name")
    feedback_text = parsed.get("feedback_text")

    print(f"✅ {parsed.get('action_text')}")

    # === Intent Routing ===
    if intent == "get_status" and name:
        return llm_query_patient(name, memory, llm_model, full_report=False)

    elif intent == "get_full_report" and name:
        return llm_query_patient(name, memory, llm_model, full_report=True)

    elif intent == "get_critical_patients":
        return get_critical_patients(memory)

    elif intent == "submit_feedback" and name and feedback_text:
        submit_feedback(name, feedback_text, memory)
        save_memory(MEMORY_PATH, memory)
        return f"Feedback noted for {name}."

    elif intent == "send_referral" and name:
        patient = next((p for p in memory.values() if p["patient_name"].lower() == name.lower()), None)
        if not patient:
            return f"No records found for {name}."
        matched = find_relevant_doctor(patient, doctors)
        if matched:
            success = send_referral_email(patient, matched)
            msg = f"Referral sent to Dr. {matched['name']} ({matched['specialization']})."
            if success:
                patient["actions_taken"].append(msg)
                save_memory(MEMORY_PATH, memory)
                return f"✅ {msg}"
            else:
                return f"❌ Failed to send referral email."
        else:
            return f"⚠️ No suitable doctor found for {name}."

    else:
        return f"✅ Noted: {query}"

# =================== BACKGROUND INGESTION ===================
def ingestion_loop(memory: Dict[str, PatientState], processed: Set[str], llm_model, doctors):
    offset, batch_size = 0, 50
    while True:
        batch = fetch_all_lab_reports(batch_size, offset)
        new_patients = [p for p in batch if p.get("unique_id") not in processed]

        if new_patients:
            prompt = f"Analyze {len(new_patients)} new lab reports. Identify risks and doctor recommendations. Respond JSON keyed by unique_id."
            try:
                assessments = json.loads(safe_llm_invoke(prompt, llm_model) or "{}")
            except json.JSONDecodeError:
                assessments = {}

            with MEMORY_LOCK:
                for data in new_patients:
                    pid, name = data.get("unique_id"), data.get("name", "Unknown")
                    assessment = assessments.get(pid, {})
                    flagged = assessment.get("risks", [])
                    doctors_rec = assessment.get("recommended_doctors", [])

                    prev = memory.get(pid)
                    labs = (prev["labs"] + [data]) if prev else [data]
                    flagged = list(set((prev["flagged_risks"] if prev else []) + flagged))
                    actions = prev["actions_taken"] if prev else []
                    feedback = prev.get("feedback", []) if prev else []

                    patient_state: PatientState = {
                        "patient_id": pid,
                        "patient_name": name,
                        "labs": labs,
                        "flagged_risks": flagged,
                        "summary": generate_summary({"patient_name": name, "labs": labs, "flagged_risks": flagged, "summary": "", "actions_taken": actions, "feedback": feedback}),
                        "actions_taken": actions,
                        "feedback": feedback
                    }

                    # === AUTO REFERRAL ===
                    if flagged:
                        matched = find_relevant_doctor(patient_state, doctors)
                        if matched:
                            try:
                                result = send_referral_email(patient_state, matched)
                                msg = f"Referral sent to Dr. {matched['name']} ({matched['specialization']})"
                                if msg not in patient_state["actions_taken"]:
                                    patient_state["actions_taken"].append(msg)
                                print(f"📩 {msg} | {result}")
                            except Exception as e:
                                print(f"⚠️ Referral error for {name}: {e}")

                    for doc in doctors_rec:
                        msg = f"Recommended doctor: {doc}"
                        if msg not in patient_state["actions_taken"]:
                            patient_state["actions_taken"].append(msg)

                    memory[pid] = patient_state
                    processed.add(pid)
                save_memory(MEMORY_PATH, memory)
            offset += batch_size
        else:
            offset = 0
        time.sleep(5)

# =================== MAIN ===================
if __name__ == "__main__":
    llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    doctors = fetch_all_doctors()
    memory: Dict[str, PatientState] = load_memory(MEMORY_PATH)
    processed: Set[str] = set(memory.keys())

    threading.Thread(target=ingestion_loop, args=(memory, processed, llm_model, doctors), daemon=True).start()

    try:
        while True:
            query = input("\nQuery (or exit): ").strip()
            if query.lower() == "exit":
                break
            print(handle_query(query, memory, llm_model, doctors))
    finally:
        save_memory(MEMORY_PATH, memory)
