"""Define the state structures for the patient monitoring agent."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Dict, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Input state representing the external interface for the patient monitoring agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """
    Tracks the primary execution state of the agent.
    
    Typical message flow:
    1. HumanMessage - doctor or clinician input
    2. AIMessage with .tool_calls - agent decides which tool(s) to use (e.g., fetch labs, update memory)
    3. ToolMessage(s) - responses from executed tools or errors
    4. AIMessage without .tool_calls - agent's final response to the user
    5. HumanMessage - next conversational input
    
    Steps 2-5 may repeat multiple times during reasoning and acting cycles.
    """


@dataclass
class State(InputState):
    """Complete agent state including ReAct lifecycle and patient-specific tracking."""

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates if the current step is the last before the agent raises a recursion error.
    Managed by the graph runtime, not manually set by user code.
    """

    # ===== Patient query tracking =====
    patient_name: str = field(default="")
    """The patient currently being queried or acted upon."""

    full_report_requested: bool = field(default=False)
    """Whether the user requested a full patient report or just a concise update."""

    # ===== Tool interaction state =====
    tools_called: List[str] = field(default_factory=list)
    """List of tool names invoked by the agent in the current reasoning cycle."""

    tool_results: Dict[str, str] = field(default_factory=dict)
    """Stores output from each tool call keyed by tool name."""

    # ===== Memory & feedback tracking =====
    patient_memory_snapshot: Dict[str, Dict] = field(default_factory=dict)
    """Snapshot of the patient's current memory state for reference during reasoning."""

    feedback_queue: List[str] = field(default_factory=list)
    """Any clinician feedback collected during the session, to be processed or stored."""

    # ===== Internal reasoning metadata =====
    recursion_depth: int = field(default=0)
    """Counts how many reasoning steps have been executed in the current cycle."""

    max_recursion_depth: int = field(default=5)
    """Maximum allowed reasoning steps before stopping to prevent infinite loops."""
