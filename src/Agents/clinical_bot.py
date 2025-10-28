# src/agents/clinical_bot.py
from typing import List, Optional
from pydantic import BaseModel
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from prompts import SYSTEM_PROMPT
from schemas import LabFlags, PatientSummary, ReferralResult
from data_agent import fetch_data_from_db
from lab_agent import analyze_patient_labs
from summarizer_agent import generate_patient_summary_func
from referral_agent import process_referral_func
import json
import re

# ----------------------------
# Pydantic Models
# ----------------------------
class PatientLabResult(BaseModel):
    patient_name: str
    flags: LabFlags
    explanation: str

# ----------------------------
# Extract patient name from query
# ----------------------------
def extract_patient_name(query: str) -> Optional[str]:
    match = re.search(r"for ([a-zA-Z\s]+)", query, re.I)
    if match:
        return match.group(1).strip()
    return None

# ----------------------------
# Tool functions
# ----------------------------
def _fetch_patient_data(name: str = None, gender: str = None, unique_id: str = None, **kwargs) -> List[dict]:
    print(f"[LOG] Fetching patient data: name={name}, gender={gender}, id={unique_id}")
    records = fetch_data_from_db(name=name, gender=gender, unique_id=unique_id, limit=10)  # fetch multiple visits
    print(f"[LOG] Fetched {len(records)} record(s)")
    return records

def _generate_patient_report(patient_records: List[dict]) -> PatientSummary:
    patient = patient_records[0]
    print(f"[LOG] Generating lab report for {patient['name']}")

    # ----------------------------
    # Reasoning Agent: Analyze Labs & Detect Flags
    # ----------------------------
    lab_results = []
    for record in patient_records:
        lab_result = analyze_patient_labs(record["name"], _fetch_patient_data)
        lab_results.append(lab_result)

    # Flatten lab flags and explanations for summarization agent
    agent_output_for_summary = []
    for i, lab_result in enumerate(lab_results):
        lab_flags_dict = lab_result.get("flags", {})
        agent_output_for_summary.append({
            "agent": f"LabAgent_Visit_{i+1}",
            "flags": lab_flags_dict,
            "explanation": lab_result.get("explanation", "")
        })

    # ----------------------------
    # Summarization Agent: Generate Patient Summary
    # ----------------------------
    summary_data = generate_patient_summary_func(
        patient_id=patient["unique_id"],
        patient_name=patient["name"],
        agent_outputs=agent_output_for_summary
    )

    # ----------------------------
    # Build structured PatientSummary
    # ----------------------------
    report = PatientSummary(
        patient_id=patient["unique_id"],
        name=patient["name"],
        summary=summary_data.get("summary", ""),
        referral_draft=summary_data.get("referral_draft", ""),
        raw_model_output=summary_data.get("raw_model_output", ""),
        explanation="\n".join([r.get("explanation", "") for r in lab_results])
    )
    print(f"[LOG] PatientSummary created: {report.model_dump()}")
    return report

def _handle_referral(patient_records: List[dict]) -> ReferralResult:
    print("[LOG] Handling referral...")
    report = _generate_patient_report(patient_records)
    referral = process_referral_func(
        patient_name=report.name,
        summary=report.summary,
        explanation=report.explanation,
        referral_draft=report.referral_draft,
        fetch_data_func=_fetch_patient_data
    )
    result = ReferralResult(**referral)
    print(f"[LOG] ReferralResult: {result.model_dump()}")
    return result

# ----------------------------
# Wrap tools for agent
# ----------------------------
fetch_tool = tool(_fetch_patient_data, description="Fetch patient data from database")
report_tool = tool(_generate_patient_report, description="Generate patient lab report and summary")
referral_tool = tool(_handle_referral, description="Generate referral instructions if needed")

# Optional JSON parsing tool to avoid Groq errors
def parse_json_tool(request: str) -> dict:
    try:
        return json.loads(request)
    except Exception as e:
        return {"error": f"JSON parse failed: {str(e)}"}

json_tool = tool(parse_json_tool, description="Parse a JSON string safely")

# ----------------------------
# LLM and Agent setup
# ----------------------------
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
agent = create_agent(
    model=llm,
    tools=[fetch_tool, report_tool, referral_tool, json_tool],
    system_prompt=SYSTEM_PROMPT
)

# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    print("[Bot] Autonomous Clinical ReAct Agent Ready. Type 'exit' to quit.")

    while True:
        user_input = input("\n🧑‍⚕️ Enter your query: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("[Bot] Goodbye!")
            break

        # Extract patient name first
        patient_name = extract_patient_name(user_input)
        if not patient_name:
            print("[Bot] ⚠️ Could not extract patient name from query.")
            continue

        print(f"[LOG] Detected patient: {patient_name}")

        # Fetch data and generate report automatically
        records = _fetch_patient_data(name=patient_name)
        if not records:
            print(f"[Bot] ⚠️ No records found for patient '{patient_name}'.")
            continue

        try:
            report = _generate_patient_report(records)
            referral = _handle_referral(records)
        except Exception as e:
            print(f"[ERROR] Failed to generate report or referral: {e}")
            continue

        # Compile structured JSON response
        structured_output = {
            "patient_summary": report.model_dump(),
            "referral_result": referral.model_dump()
        }

        print("\n[Bot Response]:", json.dumps(structured_output, indent=2))
