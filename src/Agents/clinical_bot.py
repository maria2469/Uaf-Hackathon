# src/Agents/clinical_bot_agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from Agents.data_agent import DataAgent
from Agents.lab_agent import LabInterpretationAgent
from Agents.summarizer_agent import SummarizationAgent
from Agents.referral_agent import ReferralCoordinatorAgent
import json
from dotenv import load_dotenv
load_dotenv()

class ClinicalAssistantBotAgent:
    """
    High-level orchestrator bot for clinical interaction.
    Handles:
      - Patient data queries
      - Report generation
      - Referral suggestions (with human approval)
    """

    def __init__(self):
        self.data_agent = DataAgent()
        self.lab_agent = LabInterpretationAgent(self.data_agent)
        self.summary_agent = SummarizationAgent()
        self.referral_agent = ReferralCoordinatorAgent(self.data_agent)
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

    # --------------------------------------------------------
    # INTENT DETECTION (LLM-driven)
    # --------------------------------------------------------
    def classify_intent(self, user_input: str) -> str:
        template = PromptTemplate(
            input_variables=["user_input"],
            template=(
                "Classify this request into one of the following intents:\n"
                "- info_request (fetch patient data)\n"
                "- report_request (generate patient summary)\n"
                "- referral_request (determine/send referral)\n\n"
                "User input: {user_input}\n"
                "Output only the intent name."
            )
        )
        chain = template | self.llm
        result = chain.invoke({"user_input": user_input})
        return result.content.strip().lower()

    # --------------------------------------------------------
    # HANDLERS
    # --------------------------------------------------------
    def handle_info_request(self, query_text: str):
        """Fetch patient data based on text query."""
        df = self.data_agent.fetch_by_query(query_text)
        if df.empty:
            return "No matching data found."
        return df.to_dict(orient="records")

    def handle_report_request(self, patient_id: str):
        """Generate comprehensive report for a patient."""
        lab_result = self.lab_agent.analyze(patient_id)
        summary = self.summary_agent.generate_patient_summary(patient_id, [lab_result])
        return {
            "patient_id": patient_id,
            "flags": lab_result.get("flags"),
            "explanation": lab_result.get("explanation"),
            "summary": summary.get("summary"),
            "referral_draft": summary.get("referral_draft")
        }

    def handle_referral_request(self, patient_id: str):
        """Evaluate referral necessity and handle human-in-the-loop."""
        report = self.handle_report_request(patient_id)
        summary = report["summary"]
        explanation = report["explanation"]
        referral_draft = report.get("referral_draft", "")
        referral = self.referral_agent.process_referral(patient_id, summary, explanation, referral_draft)
        return referral

    # --------------------------------------------------------
    # MAIN BOT ENTRY
    # --------------------------------------------------------
    def run(self, user_input: str):
        intent = self.classify_intent(user_input)
        print(f"[Bot] Detected intent: {intent}")

        if intent == "info_request":
            return self.handle_info_request(user_input)
        elif intent == "report_request":
            pid = self._extract_patient_id(user_input)
            return self.handle_report_request(pid)
        elif intent == "referral_request":
            pid = self._extract_patient_id(user_input)
            return self.handle_referral_request(pid)
        else:
            return "Sorry, I didn’t understand that request."

    # Simple ID extractor (can be replaced with LLM parsing)
    def _extract_patient_id(self, text: str) -> str:
        import re
        m = re.search(r'\b\d+\b', text)
        return m.group(0) if m else "unknown"


if __name__ == "__main__":
    bot = ClinicalAssistantBotAgent()
    print("[Bot] Ready to assist.")

    while True:
        q = input("\n🧑‍⚕️ Ask something (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = bot.run(q)
        print("\n[Result]", json.dumps(result, indent=2))
