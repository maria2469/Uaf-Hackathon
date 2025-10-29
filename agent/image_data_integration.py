import os, json
from threading import Lock
from typing_extensions import Dict

from agent.agent import PatientState

MEMORY_LOCK = Lock()
IMAGE_DATA_PATH = "image_data.json"

def load_image_data(path: str) -> Dict[str, Dict]:
    """Load patient data extracted from image analysis (OCR/parsing)."""
    if not os.path.exists(path):
        print(f"⚠️ No image_data.json found at {path}")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading image_data.json: {e}")
        return {}

def integrate_image_data(image_data: Dict[str, Dict], memory: Dict[str, PatientState]):
    """
    Merge image-extracted patient data into memory.
    UID is treated as the absolute unique identifier — no name/id collision issues.
    """
    with MEMORY_LOCK:
        for uid, data in image_data.items():
            if not uid or not isinstance(data, dict):
                continue  # skip invalid entries

            patient_info = data.get("patient_info", {})
            test_results = data.get("test_results", {})
            remarks = data.get("remarks", {})

            # UID = universal unique key (used even if names/patient_ids overlap)
            patient_state: PatientState = {
                "patient_id": uid,  # force UID as id
                "patient_name": patient_info.get("name", f"Unknown_{uid}"),
                "labs": [{"uid": uid, "results": test_results}],
                "flagged_risks": [],
                "summary": remarks.get("summary", "") or "No summary available.",
                "actions_taken": [],
                "feedback": []
            }

            if uid in memory:
                existing = memory[uid]
                # Merge labs
                existing_labs = existing.get("labs", [])
                existing_labs.append({"uid": uid, "results": test_results})
                existing["labs"] = existing_labs

                # Merge summaries
                if remarks.get("summary"):
                    existing["summary"] = remarks["summary"]

                # Optionally append notes or feedback (not overwrite)
                existing["actions_taken"] = list(set(existing.get("actions_taken", []) + patient_state["actions_taken"]))
                existing["flagged_risks"] = list(set(existing.get("flagged_risks", []) + patient_state["flagged_risks"]))
            else:
                memory[uid] = patient_state

        print(f"✅ Integrated {len(image_data)} image-based patient entries (UID-based).")
