import ast
import logging
import sqlite3
import smtplib
from email.message import EmailMessage
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
load_dotenv()
# ==============================
# CONFIG & SETUP
# ==============================
DB_PATH = "healthcare.db"
logger = logging.getLogger(__name__)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # e.g., clinic.bot@gmail.com
SENDER_PASS = os.getenv("SENDER_PASS")    # App-specific password (not raw Gmail password)

# ==============================
# RISK → SPECIALTY MAPPING
# ==============================
# Helps the system map flagged risks to the correct doctor domain
RISK_TO_SPECIALTY = {
    "cardiovascular risk": ["cardiology", "cardiovascular", "hypertension", "bp", "heart"],
    "respiratory risk": ["pulmonology", "respiratory", "lungs", "asthma", "infection"],
    "psychiatric risk": ["psychiatry", "mental health", "psychology", "depression", "anxiety"],
    "rheumatological risk": ["rheumatology", "arthritis", "autoimmune"],
    "hematology risk": ["hematology", "blood", "cbc", "hemoglobin"],
    "infection risk": ["infection", "infectious disease", "pathology"],
}

# ==============================
# DB HELPERS
# ==============================
def fetch_all_doctors() -> List[Dict]:
    """Fetch all doctors from the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors")
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        doctors = [dict(zip(cols, row)) for row in rows]
        return doctors
    except Exception as e:
        logger.error(f"DB error fetching doctors: {e}", exc_info=True)
        return []
    finally:
        if conn:
            conn.close()

# ==============================
# DOCTOR MATCHING
# ==============================
def find_relevant_doctor(patient_state: Dict, doctors: List[Dict]) -> Optional[Dict]:
    """
    Match the most relevant doctor for the given patient based on flagged risks or summary.
    Uses semantic mapping from risks to specialties.
    """
    if not doctors:
        logger.warning("No doctors available in database.")
        return None

    # Collect relevant keywords from patient state
    patient_keywords = set()
    flagged_risks = [r.lower() for r in patient_state.get("flagged_risks", [])]

    for risk in flagged_risks:
        patient_keywords.update(RISK_TO_SPECIALTY.get(risk, []))
    if summary := patient_state.get("summary"):
        patient_keywords.update([w.lower().strip(".,") for w in summary.split()])

    best_match, max_score = None, 0

    for doc in doctors:
        try:
            doc_keywords = set(map(str.lower, ast.literal_eval(doc.get("lab_specialty_keywords", "[]"))))
            score = len(patient_keywords & doc_keywords)

            # Prefer doctors with matching specialties and available slots
            if score > max_score and doc["current_patient_count"] < doc["max_patients_per_day"]:
                best_match, max_score = doc, score
        except Exception as e:
            logger.warning(f"Bad keyword format for doctor {doc.get('name')}: {e}")

    if best_match:
        logger.info(f"Matched doctor {best_match['name']} with score {max_score}")
    else:
        logger.warning(f"No suitable doctor found for {patient_state.get('patient_name')}")

    return best_match

# ==============================
# EMAIL REFERRAL FUNCTION
# ==============================
def send_referral_email(patient: Dict, doctor: Dict) -> bool:
    """
    Sends a structured, professional referral email with clean formatting.
    """
    if not SENDER_EMAIL or not SENDER_PASS:
        logger.error("Missing email credentials in environment variables.")
        return False

    if not doctor.get("contact_email"):
        logger.error(f"No contact email for doctor {doctor.get('name')}.")
        return False

    # Clean doctor name to avoid "Dr. Dr." issue
    doctor_name = doctor.get("name", "").strip()
    if doctor_name.lower().startswith("dr. "):
        doctor_name = doctor_name[4:].strip()

    # Format summary properly (handles multiline AI-generated summaries)
    raw_summary = patient.get("summary", "No summary available")
    clean_summary = raw_summary.strip().replace("**", "").replace("*", "")
    clean_summary = "\n".join([line.strip() for line in clean_summary.splitlines() if line.strip()])

    subject = f"Referral: Patient {patient.get('patient_name')} Requires Your Expertise"

    body = f"""
Dear Dr. {doctor_name},

A patient under monitoring has been flagged for potential clinical risk and requires your review.

───────────────────────────────
👤 **Patient Details**
───────────────────────────────
• **Name:** {patient.get('patient_name')}
• **Patient ID:** {patient.get('patient_id')}
• **Flagged Risks:** {', '.join(patient.get('flagged_risks', [])) or 'None'}

───────────────────────────────
📋 **AI-Generated Clinical Summary**
───────────────────────────────
{clean_summary}

───────────────────────────────
📞 **Next Steps**
───────────────────────────────
Please log in to the clinical dashboard or contact the care team for additional information or to take over this case.

───────────────────────────────
Best regards,  
**Automated Clinical Referral System**  
(Do not reply to this email)
"""

    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = doctor["contact_email"]
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASS)
            server.send_message(msg)
        logger.info(f"✅ Referral email sent to {doctor['name']} ({doctor['contact_email']})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send referral email: {e}", exc_info=True)
        return False
