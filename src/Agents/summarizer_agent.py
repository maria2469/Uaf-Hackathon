from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class SummarizationAgent:
    """
    Aggregates results from multiple domain agents (Lab, Imaging, Medication, etc.)
    Produces:
      1. Clinician-facing patient summary (2–3 sentences)
      2. Referral draft (if specialist follow-up is warranted)
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 0.1):
        self.llm = ChatGroq(model=model, temperature=temperature)

        self.template = PromptTemplate(
            input_variables=["patient_id", "agent_findings"],
            template=(
                "You are a senior clinical summarization assistant.\n"
                "You receive structured findings from multiple AI sub-agents analyzing different aspects of a patient's data.\n\n"
                "Patient ID: {patient_id}\n\n"
                "Agent Findings:\n{agent_findings}\n\n"
                "Your task:\n"
                "1. Write a concise 2–3 sentence clinician summary covering all relevant domains.\n"
                "2. Identify if specialist referral or follow-up is needed, and write a referral draft if applicable.\n\n"
                "Output JSON in the format:\n"
                "{{\"summary\": \"...\", \"referral_draft\": \"...\"}}\n"
                "Keep tone professional, neutral, and concise (≤120 words total)."
            )
        )

    def generate_patient_summary(self, patient_id: str, agent_outputs: List[Dict]) -> Dict:
        """
        agent_outputs example:
        [
            {"agent": "LabInterpretationAgent", "flags": {...}, "explanation": "..."},
            {"agent": "ImagingAgent", "findings": "..."},
            {"agent": "MedicationSafetyAgent", "alerts": "..."}
        ]
        """

        # Flatten all findings into a readable text
        sections = []
        for entry in agent_outputs:
            name = entry.get("agent", "UnknownAgent")
            explanation = entry.get("explanation") or entry.get("findings") or entry.get("summary") or ""
            flags = entry.get("flags") or entry.get("alerts") or {}

            section_text = f"### {name}\nFlags: {flags}\nExplanation: {explanation}\n"
            sections.append(section_text)

        agent_findings = "\n".join(sections) if sections else "No agent data available."

        # Build LCEL chain
        chain = self.template | self.llm
        response = chain.invoke({"patient_id": patient_id, "agent_findings": agent_findings})

        raw_output = response.content.strip()

        # Parse JSON-like model output safely
        summary, referral = "", ""
        try:
            parsed = json.loads(raw_output.replace("```json", "").replace("```", ""))
            summary = parsed.get("summary", "").strip()
            referral = parsed.get("referral_draft", "").strip()
        except Exception:
            summary = raw_output.split("\n")[0].strip()
            if "referr" in raw_output.lower() or "specialist" in raw_output.lower():
                referral = raw_output.strip()

        return {
            "summary": summary,
            "referral_draft": referral,
            "raw_model_output": raw_output  # keep for debugging/logging
        }


if __name__ == "__main__":
    # Example: combine outputs from multiple domain agents
    agent_outputs = [
        {
            "agent": "LabInterpretationAgent",
            "flags": {"HbA1c_increase": {"delta": 1.1}, "Creatinine_rise": {"pct_change": 14.5}},
            "explanation": "HbA1c and creatinine levels are rising, suggesting worsening glycemic control and renal stress."
        },
        {
            "agent": "MedicationSafetyAgent",
            "alerts": {"interaction": "Metformin may need dose adjustment due to renal function"},
            "explanation": "Metformin use may need review in context of impaired kidney function."
        },
        {
            "agent": "ImagingAgent",
            "findings": "Chest X-ray normal, no signs of infection or heart failure."
        }
    ]

    summarizer = SummarizationAgent()
    result = summarizer.generate_patient_summary("john_doe", agent_outputs)
    print(result)
