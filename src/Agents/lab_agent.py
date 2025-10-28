from typing import Dict, Optional
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


class LabInterpretationAgent:
    def __init__(self, db_agent):
        self.db_agent = db_agent
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

    def detect_health_flags(self, df: pd.DataFrame) -> Dict:
        flags = {}

        # Check for diseases that are high-risk
        risky_diseases = ["Asthma", "Diabetes", "Bronchitis", "Anxiety Disorders"]
        df_risk = df[df["Disease"].isin(risky_diseases)]
        if not df_risk.empty:
            flags["HighRiskDiseases"] = df_risk["Disease"].unique().tolist()

        # Check abnormal Blood Pressure
        abnormal_bp = df[df["Blood Pressure"].str.lower().isin(["high", "low"])]
        if not abnormal_bp.empty:
            flags["AbnormalBloodPressure"] = abnormal_bp["Blood Pressure"].unique().tolist()

        # Check abnormal Cholesterol Level
        abnormal_chol = df[df["Cholesterol Level"].str.lower().isin(["high", "low"])]
        if not abnormal_chol.empty:
            flags["AbnormalCholesterol"] = abnormal_chol["Cholesterol Level"].unique().tolist()

        # Check Outcome Variable (Positive findings)
        positive_outcomes = df[df["Outcome Variable"].str.lower() == "positive"]
        if not positive_outcomes.empty:
            flags["PositiveFindings"] = positive_outcomes["Disease"].unique().tolist()

        return flags

    def explain_with_llm(self, patient_name: str, flags: Dict) -> str:
        if not flags:
            return "No abnormal findings detected for this patient."

        flags_text = "\n".join([f"{k}: {v}" for k, v in flags.items()])

        template = PromptTemplate(
            input_variables=["patient_name", "flags_text"],
            template=(
                "You are a concise clinical assistant.\n"
                "Patient Name: {patient_name}\n\n"
                "Detected findings:\n{flags_text}\n\n"
                "Write a short, accurate medical summary of what this means "
                "and include one recommended next step for the clinician."
            ),
        )

        chain = template | self.llm
        result = chain.invoke({"patient_name": patient_name, "flags_text": flags_text})
        return result.content.strip()

    def analyze(self, patient_name: str) -> Dict:
        df = self.db_agent.fetch_patient_labs(patient_name)
        if df.empty:
            return {"patient_name": patient_name, "message": "No lab records found."}

        flags = self.detect_health_flags(df)
        explanation = self.explain_with_llm(patient_name, flags)

        return {
            "patient_name": patient_name,
            "flags": flags,
            "explanation": explanation
        }
