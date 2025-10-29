import threading
import os
import sys
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Ensure parent folder is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agent.agent import (
    load_memory, save_memory, handle_query,
    ingestion_loop, fetch_all_doctors,
    MEMORY_PATH
)
from langchain_groq import ChatGroq

# ================= CONFIG =================
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mock_data/healthcare.db"))

app = FastAPI(title="MEDI-AI Backend", version="1.0")

# ================= GLOBALS =================
llm_model = None
memory = {}
processed = set()
doctors = []

# ============== SCHEMAS ==============
class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    patient_name: str
    feedback: str

class ReferralRequest(BaseModel):
    patient_name: str

# ============== STARTUP ==============
@app.on_event("startup")
def startup_event():
    global llm_model, doctors, memory, processed
    try:
        print("🚀 Initializing MEDI-AI backend...")
        llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

        print(f"📂 Loading doctors from DB at {DB_PATH}...")
        doctors = fetch_all_doctors()
        if not doctors:
            print("⚠️ No doctors found in database!")

        print(f"📂 Loading memory from {MEMORY_PATH}...")
        memory = load_memory(MEMORY_PATH)
        processed = set(memory.keys())
        print(f"✅ Loaded {len(memory)} patients into memory.")

        print("⚡ Starting background ingestion thread...")
        ingestion_thread = threading.Thread(
            target=ingestion_loop, args=(memory, processed, llm_model, doctors), daemon=True
        )
        ingestion_thread.start()
        print("✅ Background ingestion started.")
    except Exception as e:
        print("❌ Startup failed:", e)
        traceback.print_exc()

# ============== ROUTES ==============
@app.get("/")
def home():
    return {"status": "ok", "message": "MEDI-AI backend running."}

@app.post("/query")
def query_agent(req: QueryRequest):
    try:
        result = handle_query(req.query, memory, llm_model, doctors)
        return {"response": result}
    except Exception as e:
        print("❌ /query error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    from agent.agent import submit_feedback
    try:
        print(f"📝 Submitting feedback for {req.patient_name}...")
        submit_feedback(req.patient_name, req.feedback, memory)
        save_memory(MEMORY_PATH, memory)
        print(f"✅ Feedback recorded for {req.patient_name}")
        return {"message": f"Feedback recorded for {req.patient_name}"}
    except Exception as e:
        print(f"❌ /feedback error for {req.patient_name}:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients")
def list_patients():
    print(f"📄 Returning {len(memory)} patients")
    return {"patients": list(memory.values())}

@app.get("/critical")
def list_critical():
    from agent.agent import get_critical_patients
    critical_patients = get_critical_patients(memory)
    print(f"⚠️ Found {len(critical_patients)} critical patients")
    return {"critical": critical_patients}

@app.post("/referral")
def manual_referral(req: ReferralRequest):
    from agent.agent import find_relevant_doctor, send_referral_email
    try:
        print(f"📨 Processing manual referral for {req.patient_name}...")
        patient = next((p for p in memory.values() if p["patient_name"].lower() == req.patient_name.lower()), None)
        if not patient:
            print(f"❌ Patient {req.patient_name} not found in memory!")
            raise HTTPException(status_code=404, detail=f"No records for {req.patient_name}")

        matched = find_relevant_doctor(patient, doctors)
        if not matched:
            print(f"❌ No suitable doctor found for {req.patient_name}")
            raise HTTPException(status_code=404, detail="No suitable doctor found")

        success = send_referral_email(patient, matched)
        if success:
            patient.setdefault("actions_taken", []).append(
                f"Manual referral to Dr. {matched['name']} ({matched['specialization']})"
            )
            save_memory(MEMORY_PATH, memory)
            print(f"✅ Referral email sent to Dr. {matched['name']}")
            return {"message": f"Referral sent to Dr. {matched['name']}"}
        else:
            print("❌ Referral email failed")
            raise HTTPException(status_code=500, detail="Referral email failed")
    except Exception as e:
        print(f"❌ /referral error for {req.patient_name}:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============== SHUTDOWN ==============
@app.on_event("shutdown")
def shutdown_event():
    try:
        save_memory(MEMORY_PATH, memory)
        print("💾 Memory saved on shutdown.")
    except Exception as e:
        print("❌ Error saving memory on shutdown:", e)
        traceback.print_exc()
