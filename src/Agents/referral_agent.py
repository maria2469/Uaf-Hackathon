# src/agents/referral_agent.py
import smtplib
from email.mime.text import MIMEText
import os
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from schemas import ReferralResult

load_dotenv()

class ReferralAgent:
    """
    Referral Agent: Determines if patient needs specialist referral and sends email to relevant doctor.
    Integrates with DataAgent to fetch doctor information and patient history.
    """
    
    def __init__(self, data_agent):
        self.data_agent = data_agent
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
        print("[ReferralAgent] ✅ Initialized")
    
    def process_referral(self, patient_name: str, summary: str, explanation: str, referral_draft: str = "") -> dict:
        """
        Main referral processing method that determines if referral is needed and sends email.
        """
        return process_referral_func(
            patient_name=patient_name,
            summary=summary,
            explanation=explanation,
            referral_draft=referral_draft,
            fetch_data_func=self.data_agent.get_all
        )

def process_referral_func(
    patient_name: str,
    summary: str,
    explanation: str,
    referral_draft: str = "",
    fetch_data_func=None
) -> dict:
    """Functional referral workflow with structured output"""

    # Step 1: Determine referral need via LLM
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
    template = PromptTemplate(
        input_variables=["summary", "explanation"],
        template=(
            "You are a clinical decision support assistant.\n"
            "Review the following patient case:\n\n"
            "Summary:\n{summary}\n\n"
            "Detailed Explanation:\n{explanation}\n\n"
            "Decide if the patient requires a specialist referral.\n"
            "If yes, specify the relevant specialty and reason.\n\n"
            "Return JSON:\n"
            '{{"refer_needed": true/false, "specialist": "...", "reason": "..."}}'
        ),
    )

    chain = template | llm
    response = chain.invoke({"summary": summary, "explanation": explanation})
    raw = response.content.strip()

    try:
        decision = json.loads(raw.replace("```json", "").replace("```", ""))
    except Exception:
        decision = {"refer_needed": False, "specialist": None, "reason": "Could not parse LLM output"}

    if not decision.get("refer_needed", False):
        return ReferralResult(
            status="no_referral_needed",
            reason=decision.get("reason"),
            specialist=None,
            doctor=None
        ).model_dump()

    specialist = (decision.get("specialist") or "").strip()
    reason = decision.get("reason") or ""

    # Step 2: Lookup doctor
    if fetch_data_func is None:
        return ReferralResult(
            status="error",
            reason="No fetch_data_func provided",
            specialist=specialist,
            doctor=None
        ).model_dump()

    doctors = fetch_data_func(table="doctors")
    if isinstance(doctors, list):
        doctors = pd.DataFrame(doctors)

    target = doctors[doctors["specialization"].str.lower().str.contains(specialist.lower(), na=False)]
    if target.empty:
        return ReferralResult(
            status="no_doctor_found",
            specialist=specialist,
            reason=reason,
            doctor=None
        ).model_dump()

    doctor = target.iloc[0]

    # Step 3: Generate referral letter/email
    email_body = f"""
Subject: Patient Referral — {patient_name} ({specialist.title()})

Dear Dr. {doctor['name']},

Please review the following patient case:

Patient Name: {patient_name}

Summary:
{summary}

Referral Note:
{referral_draft or summary}

Reason for Referral:
{reason}

Regards,
AI Clinical Assistant
"""

    # Step 4: Human-in-the-loop approval
    print("\n[ReferralAgent] 🧍 HUMAN APPROVAL REQUIRED:")
    print(f"Recommended Specialist: {specialist.title()}")
    print(f"Reason: {reason}\n")
    print("Referral Letter Preview:\n", email_body)
    confirm = input("Approve and send referral email? (y/n): ").strip().lower()

    if confirm != "y":
        return ReferralResult(
            status="pending_approval",
            specialist=specialist,
            reason=reason,
            doctor=doctor["name"]
        ).model_dump()

    # Step 5: Send email
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")

    if not email_user or not email_pass:
        return ReferralResult(
            status="send_failed",
            reason="Missing SMTP credentials",
            specialist=specialist,
            doctor=doctor["name"]
        ).model_dump()

    try:
        msg = MIMEText(email_body)
        msg["From"] = email_user
        msg["To"] = doctor["contact_email"]
        msg["Subject"] = "AI Referral Recommendation"

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)

        return ReferralResult(
            status="sent",
            doctor=doctor["name"],
            specialist=specialist,
            reason=reason
        ).model_dump()

    except Exception as e:
        return ReferralResult(
            status="send_failed",
            reason=str(e),
            specialist=specialist,
            doctor=doctor["name"]
        ).model_dump()
