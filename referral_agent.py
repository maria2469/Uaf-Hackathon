# referral_agent.py
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional

DB_PATH = "healthcare.db"  # Path to your SQLite database

# ====== Fetch doctor based on risk ======
def fetch_doctor_for_risk(risk_keyword: str) -> Optional[Dict]:
    """
    Fetch the first available doctor whose lab_specialty_keywords
    match the risk_keyword.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors")
    doctors = []
    for row in cursor.fetchall():
        cols = [col[0] for col in cursor.description]
        doc = dict(zip(cols, row))
        try:
            keywords = eval(doc.get("lab_specialty_keywords", "[]"))
            if any(risk_keyword.lower() in k.lower() for k in keywords):
                doctors.append(doc)
        except:
            continue
    conn.close()
    return doctors[0] if doctors else None

# ====== Mock Email Sender ======
def send_email(to_email: str, subject: str, body: str):
    """
    Send a dummy email. This uses a fake SMTP server for testing.
    Replace SMTP config with real one in production.
    """
    # Dummy SMTP config
    smtp_host = "smtp.example.com"
    smtp_port = 587
    smtp_user = "dummy@example.com"
    smtp_pass = "password123"

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        print(f"📨 Email sent to {to_email} (dummy). Subject: {subject}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ====== Main Referral Function ======
def auto_schedule_referral(patient_id: str, risk: str, doctor: Optional[Dict] = None) -> str:
    """
    Schedule referral for a patient given a risk and doctor.
    Returns a string describing the action.
    """
    if not doctor:
        doctor = fetch_doctor_for_risk(risk)
    if not doctor:
        action_msg = f"No doctor found for risk '{risk}' for patient {patient_id}."
        print(action_msg)
        return action_msg

    # Create referral message
    referral_msg = f"Patient {patient_id} flagged for {risk}. Please review and follow up."

    # Send dummy email
    send_email(to_email=doctor['contact_email'],
               subject=f"Referral: Patient {patient_id} - {risk}",
               body=referral_msg)

    action_msg = f"Referral scheduled for patient {patient_id} with Dr. {doctor['name']} ({risk})."
    print(f"✅ {action_msg}")
    return action_msg

# ====== Standalone Test ======
if __name__ == "__main__":
    # Example usage
    patient_id = "P001"
    risk = "High Cholesterol"
    doctor = fetch_doctor_for_risk(risk)
    auto_schedule_referral(patient_id, risk, doctor)
