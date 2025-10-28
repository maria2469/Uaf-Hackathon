# src/Agents/referral_agent.py
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Optional
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()


class ReferralCoordinatorAgent:
    """
    Handles intelligent referral workflow:
      1. Determines if referral is clinically necessary (using LLM reasoning)
      2. Identifies correct specialist based on findings
      3. Requests human approval before sending referral email
    """

    def __init__(self, db_agent):
        self.db_agent = db_agent
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_pass = os.getenv("EMAIL_PASS")

    # ---------------------------------------------------------------------
    # STEP 1 — REASONING: Determine if referral is needed
    # ---------------------------------------------------------------------
    def evaluate_referral_need(self, summary: str, explanation: str) -> Dict:
        """Use LLM reasoning to decide whether referral is justified."""
        template = PromptTemplate(
            input_variables=["summary", "explanation"],
            template=(
                "You are a clinical decision support assistant.\n"
                "Review the following patient context:\n\n"
                "Summary:\n{summary}\n\n"
                "Detailed Explanation:\n{explanation}\n\n"
                "Decide whether the patient needs a specialist referral.\n"
                "If yes, specify the most relevant specialty and why.\n\n"
                "Output JSON:\n"
                "{{\"refer_needed\": true/false, "
                "\"specialist\": \"...\", "
                "\"reason\": \"...\"}}\n"
            ),
        )

        chain = template | self.llm
        response = chain.invoke({"summary": summary, "explanation": explanation})
        raw = response.content.strip()

        try:
            result = json.loads(raw.replace("```json", "").replace("```", ""))
        except Exception:
            result = {"refer_needed": False, "specialist": None, "reason": "Could not parse LLM output"}

        return {
            "refer_needed": bool(result.get("refer_needed", False)),
            "specialist": result.get("specialist", "").strip().lower() if result.get("specialist") else None,
            "reason": result.get("reason", "").strip()
        }

    # ---------------------------------------------------------------------
    # STEP 2 — Generate referral email
    # ---------------------------------------------------------------------
    def generate_email(
        self,
        patient_name: str,
        summary: str,
        referral_draft: str,
        specialist: str,
        doctor_email: str
    ) -> str:
        """Create formatted referral email body."""
        return f"""
Subject: Patient Referral — {patient_name} ({specialist.title()})

Dear {specialist.title()} Specialist,

Please review the following patient case:

Patient Name: {patient_name}

Summary:
{summary}

Referral Note:
{referral_draft}

Regards,
AI Clinical Assistant
"""

    # ---------------------------------------------------------------------
    # STEP 3 — Send email
    # ---------------------------------------------------------------------
    def send_email(self, to_email: str, body: str) -> bool:
        """Send referral email via SMTP."""
        if not self.email_user or not self.email_pass:
            print("[ReferralAgent] ⚠️ Missing EMAIL_USER or EMAIL_PASS — email not sent.")
            return False

        try:
            msg = MIMEText(body)
            msg["From"] = self.email_user
            msg["To"] = to_email
            msg["Subject"] = "AI Referral Recommendation"

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                server.send_message(msg)
            print(f"[ReferralAgent] 📧 Email successfully sent to {to_email}")
            return True
        except Exception as e:
            print(f"[ReferralAgent] ❌ Failed to send email: {e}")
            return False

    # ---------------------------------------------------------------------
    # STEP 4 — Human-in-the-loop workflow
    # ---------------------------------------------------------------------
    def process_referral(
        self,
        patient_name: str,
        summary: str,
        explanation: str,
        referral_draft: Optional[str] = ""
    ) -> Dict:
        """Main entry point for referral workflow."""
        decision = self.evaluate_referral_need(summary, explanation)
        if not decision["refer_needed"]:
            return {"status": "no_referral_needed", "reason": decision["reason"]}

        specialist = decision["specialist"]
        reason = decision["reason"]

        # Retrieve doctor table
        doctors = self.db_agent.get_doctors()

        # Match specialist with 'specialization' field in doctors table
        target = doctors[doctors["specialization"].str.lower().str.contains(specialist, na=False)]

        if target.empty:
            return {"status": "no_doctor_found", "specialist": specialist, "reason": reason}

        doctor = target.iloc[0]
        email_body = self.generate_email(
            patient_name=patient_name,
            summary=summary,
            referral_draft=referral_draft or summary,
            specialist=specialist,
            doctor_email=doctor["contact_email"]
        )

        # --- HUMAN-IN-THE-LOOP CONFIRMATION ---
        print("\n[ReferralAgent] 🧍 HUMAN APPROVAL REQUIRED:")
        print(f"Recommended Specialist: {specialist.title()}")
        print(f"Reason: {reason}\n")
        print("Referral Email Preview:\n", email_body)
        confirm = input("Send referral email? (y/n): ").strip().lower()

        if confirm == "y":
            success = self.send_email(doctor["contact_email"], email_body)
            return {
                "status": "sent" if success else "send_failed",
                "doctor": doctor["name"],
                "specialist": specialist,
                "reason": reason
            }
        else:
            print("[ReferralAgent] ⏸️ Referral pending human approval.")
            return {"status": "pending_approval", "specialist": specialist, "reason": reason}
