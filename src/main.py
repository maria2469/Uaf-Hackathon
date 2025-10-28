# src/main.py
from pathlib import Path
from Agents.orchestrator import Coordinator
from Agents.data_agent import DataAgent

# --- Paths ---
BASE = Path(__file__).parent
DB_PATH = BASE / "mock_data" / "healthcare.db"

def demo():
    # 1️⃣ Ensure database exists
    if not DB_PATH.exists():
        raise FileNotFoundError(f"❌ Database not found at {DB_PATH}. Please add healthcare.db in /mock_data.")

    print(f"[main] ✅ Using existing database: {DB_PATH}")

    # 2️⃣ Initialize system coordinator (handles agents + workflow)
    coord = Coordinator(db_path=DB_PATH)

    # 3️⃣ Initialize DataAgent for direct queries & table introspection
    data_agent = coord.data_agent  # Use shared DataAgent from Coordinator

    # 4️⃣ List available tables
    available_tables = data_agent.list_tables()
    print(f"[main] 📋 Available tables: {available_tables}")

    # Ensure 'lab_reports' exists
    if "lab_reports" not in available_tables:
        raise ValueError("❌ Table 'lab_reports' not found in the database!")

    # 5️⃣ Extract unique patients
    lab_df = data_agent.get_all("lab_reports")
    patient_column = "name" if "name" in lab_df.columns else "unique_id"
    patients = lab_df[patient_column].dropna().unique().tolist()
    print(f"[main] 👩‍⚕️ Found {len(patients)} patients in DB: {patients}")

    # 6️⃣ Run orchestrator workflow for each patient
    for patient_id in patients:
        result = coord.run_patient_workflow(patient_id)

        print("\n====================================")
        print(f"🧠 Patient ID: {patient_id}")
        print("------------------------------------")
        print("🔹 Lab Flags:", result.get("lab_flags"))
        print("🔹 Explanation:", result.get("lab_explanation"))
        print("🔹 Summary:", result.get("summary"))
        print("🔹 Referral Draft:", result.get("referral_draft"))
        print("====================================\n")

if __name__ == "__main__":
    demo()
