"""Define the configurable parameters for the patient monitoring ReAct agent."""

from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from typing import Annotated

# You can create a separate prompts.py or define the prompt inline
SYSTEM_PROMPT = """
You are a clinical reasoning assistant and autonomous agent for patient health monitoring.
Your responsibilities:
1. Assess patient lab reports, medical history, and feedback.
2. Identify potential health risks and recommend doctors.
3. Suggest actions such as updating flagged risks, generating summaries, or requesting feedback.
4. Respond in JSON when interacting with tools, or readable text when answering queries.
5. Always consider previous patient feedback and actions taken.
"""

@dataclass(kw_only=True)
class Context:
    """The context configuration for the patient monitoring agent."""

    # ======= System prompt =======
    system_prompt: str = field(
        default=SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt defining the agent's behavior, role, and reasoning style."
        },
    )

    # ======= LLM Model =======
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-oss-20b",
        metadata={
            "description": "The language model used for reasoning and acting in the agent."
        },
    )

    # ======= Ingestion settings =======
    ingestion_batch_size: int = field(
        default=50,
        metadata={
            "description": "Number of lab reports to fetch per batch for background ingestion."
        },
    )

    ingestion_sleep_sec: int = field(
        default=5,
        metadata={
            "description": "Sleep time in seconds between background ingestion cycles."
        },
    )

    # ======= Memory settings =======
    memory_file: str = field(
        default="patient_memory.json",
        metadata={
            "description": "Path to the JSON file storing patient memory and state."
        },
    )

    # ======= Misc settings =======
    max_tool_attempts: int = field(
        default=3,
        metadata={
            "description": "Maximum number of times the agent should attempt tool calls in one reasoning cycle."
        },
    )

    def __post_init__(self) -> None:
        """Override defaults with environment variables if provided."""
        for f in fields(self):
            if not f.init:
                continue
            env_value = os.environ.get(f.name.upper())
            if env_value is not None:
                # Convert types correctly
                if isinstance(f.default, int):
                    env_value = int(env_value)
                setattr(self, f.name, env_value)
