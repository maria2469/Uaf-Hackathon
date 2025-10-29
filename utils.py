"""Utility & helper functions for the Patient Monitoring ReAct Agent."""

from typing import Union, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_groq import ChatGroq
import traceback


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
    """Load a chat model for the agent (ChatGroq).

    Args:
        model_name (str): Fully specified model name, e.g., 'openai/gpt-oss-20b'.
        temperature (float, optional): Sampling temperature. Defaults to 0.

    Returns:
        ChatGroq: Initialized chat model.
    """
    try:
        return ChatGroq(model=model_name, temperature=temperature)
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        raise


def safe_llm_invoke(prompt: str, llm_model: ChatGroq) -> str:
    """Safely invoke the LLM and catch errors.

    Args:
        prompt (str): Prompt text.
        llm_model (ChatGroq): Initialized LLM instance.

    Returns:
        str: LLM response text or error message.
    """
    try:
        response = llm_model.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        print(f"❌ LLM invocation error: {e}")
        traceback.print_exc()
        return "LLM unavailable"
