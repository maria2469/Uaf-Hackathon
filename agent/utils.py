"""Utility & helper functions for the Patient Monitoring ReAct Agent."""

import json
import sqlite3
import traceback
from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
# ==============================================================
# --------------------  LLM Utilities  --------------------------
# ==============================================================

def get_message_text(msg: BaseMessage) -> str:
    """Extract the textual content from a BaseMessage."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    elif isinstance(content, list):
        texts = [c if isinstance(c, str) else c.get("text", "") for c in content]
        return "".join(texts).strip()
    return ""


def load_chat_model(model_name: str, temperature: float = 0) -> ChatGroq:
    """Load a chat model for the agent (ChatGroq)."""
    try:
        return ChatGroq(model=model_name, temperature=temperature)
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        raise


def safe_llm_invoke(prompt: str, llm_model: ChatGroq) -> str:
    """Safely invoke the LLM and catch errors."""
    try:
        response = llm_model.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        print(f"❌ LLM invocation error: {e}")
        traceback.print_exc()
        return "LLM unavailable"

# ==============================================================
# --------------------  Data Utilities  -------------------------
# ==============================================================

def fetch_all_lab_reports(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch all lab reports from the SQLite database.
    Modify this according to your DB schema.
    """
    try:
        conn = sqlite3.connect("patient_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, patient_name, lab_name, result, date FROM lab_reports LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        reports = []
        for row in rows:
            reports.append({
                "patient_id": row[0],
                "patient_name": row[1],
                "lab_name": row[2],
                "result": row[3],
                "date": row[4],
            })
        return reports
    except Exception as e:
        print(f"❌ Error fetching lab reports: {e}")
        return []


def load_memory(filepath: str) -> Dict[str, Any]:
    """Load agent memory (patient summaries, state, etc.) from JSON."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ No memory file found, starting fresh.")
        return {}
    except json.JSONDecodeError:
        print("⚠️ Corrupted memory file. Starting with empty memory.")
        return {}


def save_memory(filepath: str, data: Dict[str, Any]) -> None:
    """Save memory back to JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to save memory: {e}")
