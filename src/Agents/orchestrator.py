# src/orchestrator.py
from pathlib import Path
from Agents.data_agent import DataAgent
from Agents.lab_agent import LabInterpretationAgent
from Agents.summarizer_agent import SummarizationAgent
from Agents.referral_agent import ReferralAgent


class Coordinator:
    """
    LangGraph-style orchestrator:
      Nodes: DataAgent -> LabInterpretationAgent -> SummarizationAgent
      Data flows through nodes, and all intermediate outputs are stored in a DAG-like structure.
    """

    def __init__(self, db_path: Path = None):
        # Nodes
        self.data_agent = DataAgent(db_path)
        self.lab_agent = LabInterpretationAgent(self.data_agent)
        self.summary_agent = SummarizationAgent()
        self.referral_agent = ReferralAgent(self.data_agent)

        # DAG to store outputs per node
        self.dag_data_flow = {}

        # Define node order (topological order for execution)
        self.nodes = ["data_agent", "lab_agent", "summary_agent", "referral_agent"]

    def run_patient_workflow(self, patient_id: str) -> dict:
        """
        Executes a patient workflow using a LangGraph-style DAG.
        Each node consumes input from previous nodes and produces outputs stored in dag_data_flow.
        """

        # Node: DataAgent -> fetch patient labs
        self.dag_data_flow["lab_data"] = self.data_agent.fetch_patient_labs(patient_id)

        # Node: LabInterpretationAgent -> analyze labs
        self.dag_data_flow["lab_result"] = self.lab_agent.analyze(patient_id)
        lab_result = self.dag_data_flow["lab_result"]
        flags = lab_result.get("flags", {})
        explanation = lab_result.get("explanation", "No explanation available")

        # Node: SummarizationAgent -> generate summary
        agent_outputs = [
            {
                "agent": "LabInterpretationAgent",
                "flags": flags,
                "explanation": explanation,
            }
        ]
        self.dag_data_flow["summary_package"] = self.summary_agent.generate_patient_summary(
            patient_id, agent_outputs
        )
        summary_package = self.dag_data_flow["summary_package"]

        # Node: ReferralAgent -> process referral if needed
        self.dag_data_flow["referral_result"] = self.referral_agent.process_referral(
            patient_name=patient_id,
            summary=summary_package.get("summary", ""),
            explanation=explanation,
            referral_draft=summary_package.get("referral_draft", "")
        )
        referral_result = self.dag_data_flow["referral_result"]

        # Final consolidated output
        return {
            "patient_id": patient_id,
            "lab_flags": flags,
            "lab_explanation": explanation,
            "summary": summary_package.get("summary", ""),
            "referral_draft": summary_package.get("referral_draft", ""),
            "referral_status": referral_result.get("status", ""),
            "referral_specialist": referral_result.get("specialist", ""),
            "referral_doctor": referral_result.get("doctor", ""),
            "referral_reason": referral_result.get("reason", ""),
            "raw_summary_output": summary_package.get("raw_model_output", ""),
            "dag_data_flow": self.dag_data_flow  # Expose full DAG outputs for debugging/UI
        }


if __name__ == "__main__":
    coordinator = Coordinator()
    patient_id = "Noah Brown"
    result = coordinator.run_patient_workflow(patient_id)

    print(f"\n🧠 Workflow complete for patient: {patient_id}")
    print("------------------------------------------------")
    print("Flags:", result["lab_flags"])
    print("Explanation:", result["lab_explanation"])
    print("\nSummary:", result["summary"])
    print("\nReferral Draft:", result["referral_draft"])
