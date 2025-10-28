# src/agents/summarizer_agent.py
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from schemas import PatientSummary
import json

def generate_patient_summary_func(
    patient_id: str,
    patient_name: str,
    agent_outputs: List[Dict]
) -> dict:
    """
    Summarization Agent: Generate a structured patient summary from multiple agent outputs.
    Handles trend analysis, inconsistencies, and referral suggestions.
    Returns a validated dictionary via PatientSummary Pydantic model.
    """

    print(f"[LOG] Generating patient summary for {patient_name} ({patient_id})")

    # -----------------------------
    # Aggregate agent outputs
    # -----------------------------
    sections = []
    explanations = []
    trends = []

    for idx, entry in enumerate(agent_outputs, start=1):
        agent_name = entry.get("agent", f"LabAgent_Visit_{idx}")
        explanation = entry.get("explanation") or entry.get("summary") or ""
        flags = entry.get("flags") or {}

        # Detect trends: worsening flags across visits
        trend_detected = [k for k, v in flags.items() if v]
        if trend_detected:
            trends.append(f"{agent_name}: {trend_detected}")

        section_text = f"### {agent_name}\nFlags: {flags}\nExplanation: {explanation}\n"
        sections.append(section_text)
        explanations.append(explanation)

    agent_findings = "\n".join(sections) if sections else "No agent data available."
    combined_explanation = "\n".join(explanations) if explanations else "No explanation available."
    trend_summary = "\n".join(trends) if trends else "No significant trends detected."

    # -----------------------------
    # Build prompt for LLM (corrected JSON braces)
    # -----------------------------
    prompt_text = f"""
Patient ID: {{patient_id}}
Patient Name: {{patient_name}}

Agent Findings:
{{agent_findings}}

Trend Summary:
{trend_summary}

Tasks:
1. Write a concise 3–5 sentence clinician summary covering all relevant clinical domains.
2. Identify inconsistencies, contradictions, or gaps in the patient data.
3. Recommend if specialist referral or follow-up is needed and draft referral instructions.

Output JSON strictly in this format:
{{
    "summary": "...",
    "referral_draft": "..."
}}
"""

    print("[DEBUG] Prompt text being used:\n", prompt_text)
    print("[DEBUG] Input variables: patient_id, patient_name, agent_findings")
    print("[DEBUG] Values being passed:")
    print("patient_id:", patient_id)
    print("patient_name:", patient_name)
    print("agent_findings:", agent_findings)
    print("trend_summary:", trend_summary)

    # -----------------------------
    # Invoke LLM
    # -----------------------------
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
    template = PromptTemplate(
        input_variables=["patient_id", "patient_name", "agent_findings"],
        template=prompt_text
    )
    chain = template | llm

    try:
        response = chain.invoke({
            "patient_id": patient_id,
            "patient_name": patient_name,
            "agent_findings": agent_findings
        })
    except KeyError as ke:
        print("[ERROR] KeyError in PromptTemplate invocation:", str(ke))
        raise
    except Exception as e:
        print("[ERROR] LLM invocation failed:", str(e))
        response = type('obj', (object,), {'content': '{"summary":"LLM failure","referral_draft":""}'})()

    raw_output = response.content.strip()
    summary_text = ""
    referral_draft_text = ""

    # -----------------------------
    # Parse JSON from LLM output
    # -----------------------------
    try:
        parsed = json.loads(raw_output.replace("```json", "").replace("```", ""))
        summary_text = parsed.get("summary", "").strip()
        referral_draft_text = parsed.get("referral_draft", "").strip()
    except Exception as e:
        print("[WARN] Failed to parse LLM output as JSON:", str(e))
        summary_text = raw_output.split("\n")[0].strip()
        referral_draft_text = raw_output.strip() if "referr" in raw_output.lower() else ""

    # -----------------------------
    # Build Pydantic PatientSummary
    # -----------------------------
    summary_model = PatientSummary(
        patient_id=patient_id,
        name=patient_name,
        summary=summary_text,
        referral_draft=referral_draft_text,
        raw_model_output=raw_output,
        explanation=combined_explanation
    )

    print("[LOG] PatientSummary prepared:", summary_model.model_dump())
    return summary_model.model_dump()
