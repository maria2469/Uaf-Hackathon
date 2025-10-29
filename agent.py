import time
import json
import sqlite3
from typing import List, TypedDict, Dict
from summarizer_agent import generate_patient_summary_func
from referral_agent import auto_schedule_referral
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ====== Agent State ======
class PatientState(TypedDict):
    patient_id: str
    patient_name: str
    labs: Dict
    flagged_risks: List[str]
    summary: str
    actions_taken: List[str]

# ====== Database Layer ======
DB_PATH = "healthcare.db"

def fetch_lab_report(unique_id: str) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lab_reports WHERE unique_id=?", (unique_id,))
    row = cursor.fetchone()
    if row:
        cols = [col[0] for col in cursor.description]
        data = dict(zip(cols, row))
    else:
        data = {}
    conn.close()
    return data

def fetch_doctor_for_risk(risk_keyword: str) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors")
    doctors = []
    for row in cursor.fetchall():
        cols = [col[0] for col in cursor.description]
        doc = dict(zip(cols, row))
        keywords = eval(doc.get("lab_specialty_keywords", "[]"))
        if any(risk_keyword.lower() in k.lower() for k in keywords):
            doctors.append(doc)
    conn.close()
    return doctors[0] if doctors else None

# ====== Reasoning Agent ======
def detect_risks(patient_state: PatientState, llm_model=None) -> List[str]:
    labs = patient_state["labs"]
    risks = []

    # Rule-based checks
    try:
        if labs.get("Cholesterol Level") and float(labs["Cholesterol Level"]) > 240:
            risks.append("High Cholesterol")
        if labs.get("Fever") and labs["Fever"].lower() == "yes":
            risks.append("Possible Infection")
        if labs.get("Fatigue") and labs["Fatigue"].lower() == "yes":
            risks.append("Check Anemia / Thyroid")
    except:
        pass

    # Optional LLM reasoning
    if llm_model:
        try:
            prompt = f"""
Patient Labs: {json.dumps(labs)}
Identify potential health risks in concise bullet points.
Respond ONLY in JSON: {{ "risks": ["risk1", "risk2", ...] }}
"""
            response = llm_model.invoke([{"role": "user", "content": prompt}])
            llm_risks = json.loads(response.content).get("risks", [])
            risks.extend(llm_risks)
        except Exception as e:
            print(f"❌ LLM risk detection failed: {e}")

    return list(set(risks))

# ====== Summarization Agent ======
def generate_summary(patient_state: PatientState) -> str:
    try:
        return generate_patient_summary_func(patient_state)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return "Summary unavailable."

# ====== Action / Coordination Agent ======
def take_actions(patient_state: PatientState):
    for risk in patient_state["flagged_risks"]:
        doctor = fetch_doctor_for_risk(risk)
        try:
            action_msg = auto_schedule_referral(patient_state["patient_id"], risk, doctor)
            patient_state["actions_taken"].append(action_msg)
        except Exception as e:
            print(f"❌ Failed to schedule referral for {risk}: {e}")

# ====== Main Agent Loop ======
if __name__ == "__main__":
    llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    patient_unique_id = "UID_00001"
    lab_data = fetch_lab_report(patient_unique_id)

    if not lab_data:
        print("❌ No lab data found for this patient.")
        exit()

    patient_state: PatientState = {
        "patient_id": lab_data.get("unique_id", ""),
        "patient_name": lab_data.get("name", ""),
        "labs": lab_data,
        "flagged_risks": [],
        "summary": "",
        "actions_taken": []
    }

    try:
        while True:
            # 1️⃣ Reasoning
            patient_state["flagged_risks"] = detect_risks(patient_state, llm_model=llm_model)

            # 2️⃣ Summarization
            patient_state["summary"] = generate_summary(patient_state)

            # 3️⃣ Actions
            take_actions(patient_state)

            # 4️⃣ Display / Mock Dashboard
            print(f"\nPatient: {patient_state['patient_name']}")
            print(f"Flagged Risks: {patient_state['flagged_risks']}")
            print(f"Summary: {patient_state['summary']}")
            print(f"Actions Taken: {patient_state['actions_taken']}")

            time.sleep(10)

    except KeyboardInterrupt:
        print("🛑 UPI Agent stopped by user.")
