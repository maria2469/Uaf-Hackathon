# src/agents/data_agent.py

from pathlib import Path
import pandas as pd
import sqlite3
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "mock_data" / "healthcare.db"


class DataAgent:
    """
    Unified data access layer for the healthcare agentic system.
    Handles DB queries, schema discovery, and LLM-based natural language interpretation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"❌ Database not found at {self.db_path}")

        # LLM initialization
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
        print(f"[DataAgent] ✅ Connected to {self.db_path.name}")

    # -------------------------------------------------------------------------
    # 🧠 DATABASE UTILITIES
    # -------------------------------------------------------------------------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Safely execute SQL query and return as DataFrame."""
        try:
            with self._connect() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            print(f"[DataAgent] ❌ SQL Error: {e}")
            return pd.DataFrame()

    def list_tables(self) -> list:
        df = self._query("SELECT name FROM sqlite_master WHERE type='table';")
        return df["name"].tolist() if not df.empty else []

    def list_columns(self, table: str) -> list:
        try:
            with self._connect() as conn:
                cursor = conn.execute(f"PRAGMA table_info({table});")
                return [row[1] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[DataAgent] ⚠️ Could not list columns for {table}: {e}")
            return []

    def describe_database(self):
        """Print all tables and their columns in a structured format."""
        print(f"\n[DataAgent] 🧩 Database Summary for {self.db_path}")
        for table in self.list_tables():
            cols = self.list_columns(table)
            print(f"📂 Table: {table} ({len(cols)} columns)")
            print(f"   └── {cols}")
        print("")

    # -------------------------------------------------------------------------
    # 🔍 LLM QUERY INTERPRETATION
    # -------------------------------------------------------------------------
    def interpret_query(self, query_text: str) -> Dict[str, Any]:
        """
        Converts natural language into a structured intent for SQL querying.
        """
        prompt = PromptTemplate(
            input_variables=["query_text"],
            template=(
                "You are a data interpreter for a unified healthcare system.\n"
                "Convert the user's request into JSON with the following keys:\n"
                "- table: which table to query (patients, lab_reports, doctors, visits, prescriptions)\n"
                "- filters: key-value pairs for filtering.\n\n"
                "Ensure the JSON is strictly valid and uses only existing column names "
                "from the table (no hallucinations).\n\n"
                "Example:\n"
                "Input: Show all lab reports for patient John Doe\n"
                "Output: {{\"table\": \"lab_reports\", \"filters\": {{\"name\": \"John Doe\"}}}}\n\n"
                "Input: {query_text}\nOutput:"
            ),
        )

        chain = prompt | self.llm
        result = chain.invoke({"query_text": query_text})

        try:
            return json.loads(result.content.strip())
        except Exception:
            print("[DataAgent] ⚠️ LLM output invalid, using fallback defaults.")
            return {"table": "lab_reports", "filters": {}}

    # -------------------------------------------------------------------------
    # 🧾 QUERY EXECUTION
    # -------------------------------------------------------------------------
    def fetch_by_query(self, query_text: str) -> pd.DataFrame:
        """Run an interpreted user query safely against the database."""
        structured = self.interpret_query(query_text)
        table = structured.get("table", "lab_reports")
        filters = structured.get("filters", {})

        if table not in self.list_tables():
            print(f"[DataAgent] ❌ Table '{table}' not found.")
            return pd.DataFrame()

        columns = self.list_columns(table)
        valid_filters = {k: v for k, v in filters.items() if k in columns}
        if len(valid_filters) < len(filters):
            print("[DataAgent] ⚠️ Some filters ignored due to invalid columns.")

        base_query = f"SELECT * FROM {table}"
        params = ()
        if valid_filters:
            conditions = [f"{col} = ?" for col in valid_filters.keys()]
            base_query += " WHERE " + " AND ".join(conditions)
            params = tuple(valid_filters.values())

        df = self._query(base_query, params)
        print(f"[DataAgent] ✅ Retrieved {len(df)} records from '{table}'.")
        return df

    # -------------------------------------------------------------------------
    # 📊 DIRECT ACCESS HELPERS
    # -------------------------------------------------------------------------
    def get_all(self, table: str) -> pd.DataFrame:
        """Quick access to all data in a given table."""
        if table not in self.list_tables():
            print(f"[DataAgent] ❌ Table '{table}' does not exist.")
            return pd.DataFrame()
        return self._query(f"SELECT * FROM {table}")

    def fetch_patient_labs(self, patient_name: str) -> pd.DataFrame:
        """Fetch all lab reports for a specific patient by name."""
        if "lab_reports" not in self.list_tables():
            print("[DataAgent] ❌ 'lab_reports' table not found.")
            return pd.DataFrame()
        return self._query("SELECT * FROM lab_reports WHERE name = ?", (patient_name,))

    def fetch_patient_data(self, patient_id_or_name: str, table: str = "lab_reports", id_column: str = "name") -> pd.DataFrame:
        """
        Generic method to fetch any patient-specific data from any table.
        `id_column` defaults to 'name', can be overridden to 'unique_id'.
        """
        if table not in self.list_tables():
            print(f"[DataAgent] ❌ Table '{table}' not found.")
            return pd.DataFrame()
        return self._query(f"SELECT * FROM {table} WHERE {id_column} = ?", (patient_id_or_name,))
        

# -------------------------------------------------------------------------
# 🧪 Manual Testing
# -------------------------------------------------------------------------
if __name__ == "__main__":
    agent = DataAgent()
    print("Available Tables:", agent.list_tables())
    agent.describe_database()

    # Example fetch by patient name
    df = agent.fetch_patient_labs("Noah Brown")
    print("\nFetched Data for Noah Brown:\n", df.head())

    # Example interpreted query
    df2 = agent.fetch_by_query("Show me all lab reports for patient Noah Brown")
    print("\nFetched Data by Query:\n", df2.head())
