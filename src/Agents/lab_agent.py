# src/agents/lab_agent.py
from typing import Dict, List
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from prompts import LAB_ANALYSIS_PROMPT
from schemas import LabResult, LabFlags

def analyze_patient_labs(patient_name: str, fetch_data_func) -> Dict:
    """
    Reasoning Agent: Analyze lab data for a patient across multiple visits.
    Detects risky trends and abnormalities.
    Returns a validated LabResult dictionary.
    """

    print(f"[LOG] Analyzing labs for patient: {patient_name}")

    # -----------------------------
    # Fetch lab data
    # -----------------------------
    df = fetch_data_func(name=patient_name, gender=None, unique_id=None)
    if isinstance(df, list):
        df = pd.DataFrame(df)

    if df.empty:
        print(f"[LOG] No lab records found for {patient_name}")
        return LabResult(
            patient_name=patient_name,
            flags=LabFlags(),
            explanation="No lab records found."
        ).dict()

    # -----------------------------
    # Detect health flags
    # -----------------------------
    flags = LabFlags()

    # High-risk diseases
    risky_diseases = ["Asthma", "Diabetes", "Bronchitis", "Anxiety Disorders"]
    df_risk = df[df["Disease"].isin(risky_diseases)]
    if not df_risk.empty:
        flags.HighRiskDiseases = df_risk["Disease"].unique().tolist()

    # Abnormal blood pressure
    if "Blood Pressure" in df.columns:
        abnormal_bp = df[df["Blood Pressure"].str.lower().isin(["high", "low"])]
        if not abnormal_bp.empty:
            flags.AbnormalBloodPressure = abnormal_bp["Blood Pressure"].unique().tolist()

    # Abnormal cholesterol
    if "Cholesterol Level" in df.columns:
        abnormal_chol = df[df["Cholesterol Level"].str.lower().isin(["high", "low"])]
        if not abnormal_chol.empty:
            flags.AbnormalCholesterol = abnormal_chol["Cholesterol Level"].unique().tolist()

    # Positive outcomes
    if "Outcome Variable" in df.columns:
        positive_outcomes = df[df["Outcome Variable"].str.lower() == "positive"]
        if not positive_outcomes.empty:
            flags.PositiveFindings = positive_outcomes["Disease"].unique().tolist()

    # -----------------------------
    # LLM explanation for reasoning agent
    # -----------------------------
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

    if all(not getattr(flags, attr) for attr in flags.__fields__):
        explanation = "No abnormal findings detected for this patient."
        print(f"[LOG] {patient_name}: {explanation}")
    else:
        # Flatten flags for LLM prompt
        flags_text = "\n".join([f"{k}: {getattr(flags, k)}" for k in flags.__fields__])
        template = PromptTemplate(
            input_variables=["patient_name", "flags_text"],
            template=LAB_ANALYSIS_PROMPT
        )
        chain = template | llm
        try:
            result = chain.invoke({"patient_name": patient_name, "flags_text": flags_text})
            explanation = result.content.strip()
        except Exception as e:
            explanation = f"LLM explanation failed: {str(e)}"
            print(f"[ERROR] {patient_name}: {explanation}")

    # -----------------------------
    # Build LabResult
    # -----------------------------
    lab_result = LabResult(
        patient_name=patient_name,
        flags=flags,
        explanation=explanation
    )

    print(f"[LOG] Lab analysis complete for {patient_name}: {lab_result.dict()}")
    return lab_result.dict()
