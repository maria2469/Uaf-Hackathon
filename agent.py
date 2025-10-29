import time
import json
import sqlite3
import os
import threading
import re
from typing import Dict, List, Set, TypedDict, Optional
from langchain_groq import ChatGroq
from summarizer_agent import generate_patient_summary_func
from dotenv import load_dotenv
from difflib import get_close_matches
from threading import Lock

from utils import safe_llm_invoke

load_dotenv()

# =================== Agent State ===================
class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: List[Dict]
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]
    feedback: List[str]

DB_PATH = "healthcare.db"
MEMORY_PATH = "patient_memory.json"
MEMORY_LOCK = Lock()

# =================== DB Helpers ===================
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

def fetch_all_doctors() -> List[Dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors")
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        print(f"❌ DB error fetching doctors: {e}")
        return []
    finally:
        conn.close()

# =================== Memory Helpers ===================
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

# =================== Summarization ===================
def generate_summary(patient_state: PatientState) -> str:
    try:
        return generate_patient_summary_func(patient_state)
    except Exception as e:
        print(f"❌ Summary error: {e}")
        return "Summary unavailable."

# =================== Feedback ===================
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

# =================== Query Parsing ===================
def extract_patient_name(query: str, memory: Dict[str, PatientState]) -> Optional[str]:
    # Look for keywords first
    patterns = [
        r"(?:how is|status of|update on|show report for|full report for|medical history of)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    ]
    for pat in patterns:
        match = re.search(pat, query, re.IGNORECASE)
        if match:
            name_candidate = match.group(1).strip()
            # Fuzzy match against memory
            all_names = [p["patient_name"] for p in memory.values()]
            match_name = get_close_matches(name_candidate, all_names, n=1, cutoff=0.6)
            if match_name:
                return match_name[0]
            return name_candidate

    # Fallback: check any patient name in query
    for p in memory.values():
        if p["patient_name"].lower() in query.lower():
            return p["patient_name"]
    return None

def is_full_report(query: str) -> bool:
    return any(k in query.lower() for k in ["medical history", "full report", "detailed report", "complete report"])

def is_casual_query(query: str) -> bool:
    return any(k in query.lower() for k in ["how is", "okay", "status", "info", "update on"])

def is_critical_query(query: str) -> bool:
    return any(k in query.lower() for k in ["critical patient", "patients at risk", "who is critical", "patients with risks"])

# =================== LLM Patient Query ===================
AGENT_PROMPT = """
You are a clinical assistant AI. Your job is to respond to any patient query dynamically.
Rules:
1. Determine query type (casual, normal update, full report).
2. Always include flagged risks, latest labs, and feedback if any.
3. Include reasoning trace (steps) for normal/full queries.
4. Never hallucinate data. Use memory/DB only.
5. Provide critical patient list if asked.
"""

def llm_query_patient(name: str, memory: Dict[str, PatientState], llm_model, full_report=False, casual=False) -> str:
    patient = next((p for p in memory.values() if p["patient_name"].lower() == name.lower()), None)
    if not patient:
        return f"No records found for patient {name}"

    patient_copy = dict(patient)
    patient_copy["labs"] = [patient["labs"][-1]] if patient["labs"] else []

    record = json.dumps(patient_copy, indent=2)
    instructions = "Provide a concise summary (~4-6 lines with reasoning)."
    if full_report:
        instructions = "Provide FULL structured report including reasoning trace."
    elif casual:
        instructions = "Provide 1-2 line casual summary without reasoning."

    prompt = f"""
{AGENT_PROMPT}

User request about patient {name}.
Instructions: {instructions}

Patient record:
{record}
"""

    for attempt in range(3):
        try:
            response = safe_llm_invoke(prompt, llm_model)
            if response:
                return response
        except Exception as e:
            print(f"❌ LLM error: {e}, retrying ({attempt+1}/3)")
            time.sleep(1)

    return "LLM query failed."

# =================== Critical Patient Tool ===================
def get_critical_patients(memory: Dict[str, PatientState]) -> str:
    critical_list = []
    for p in memory.values():
        if p["flagged_risks"]:
            critical_list.append({
                "name": p["patient_name"],
                "id": p["patient_id"],
                "risk": ", ".join(p["flagged_risks"])
            })
    if not critical_list:
        return "No critical patients currently."
    return json.dumps(critical_list, indent=2)

# =================== Background Ingestion ===================
def ingestion_loop(memory: Dict[str, PatientState], processed: Set[str], llm_model, doctors):
    offset, batch_size = 0, 50
    while True:
        batch = fetch_all_lab_reports(batch_size, offset)
        new_patients = [p for p in batch if p.get("unique_id") not in processed]

        if new_patients:
            prompt = f"Assess {len(new_patients)} patients, suggest flagged risks and recommended doctors."
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

                    if pid in memory:
                        prev = memory[pid]
                        labs = prev["labs"] + [data]
                        flagged = list(set(prev["flagged_risks"] + flagged))
                        actions = prev["actions_taken"]
                        feedback = prev.get("feedback", [])
                    else:
                        labs, actions, feedback = [data], [], []

                    patient_state: PatientState = {
                        "patient_id": pid,
                        "patient_name": name,
                        "labs": labs,
                        "flagged_risks": flagged,
                        "summary": "",
                        "actions_taken": actions,
                        "feedback": feedback
                    }
                    patient_state["summary"] = generate_summary(patient_state)
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

# =================== Main Console ===================
if __name__ == "__main__":
    llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    doctors = fetch_all_doctors()
    memory: Dict[str, PatientState] = load_memory(MEMORY_PATH)
    processed: Set[str] = set(memory.keys())

    threading.Thread(target=ingestion_loop, args=(memory, processed, llm_model, doctors), daemon=True).start()

    last_patient = None
    try:
        while True:
            query = input("\nQuery (or exit): ").strip()
            if query.lower() == "exit":
                break

            if query.lower().startswith("feedback:"):
                try:
                    _, rest = query.split(":", 1)
                    if "|" in rest:
                        patient_name, fb = rest.split("|", 1)
                        submit_feedback(patient_name.strip(), fb.strip(), memory)
                        save_memory(MEMORY_PATH, memory)
                    else:
                        print("❌ Feedback must include patient name using '|', e.g., feedback: Sophia Martinez|Great care")
                except Exception as e:
                    print(f"❌ Invalid feedback: {e}")
                continue

            if is_critical_query(query):
                print(get_critical_patients(memory))
                continue

            name = extract_patient_name(query, memory)
            if name:
                last_patient = name
                full_report = is_full_report(query)
                casual = is_casual_query(query)
                print(llm_query_patient(name, memory, llm_model, full_report=full_report, casual=casual))
            else:
                print(f"✅ Noted: {query}")

    finally:
        save_memory(MEMORY_PATH, memory)
