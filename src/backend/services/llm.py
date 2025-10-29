from typing import Optional

from langchain_groq import ChatGroq


_llm: Optional[ChatGroq] = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
    return _llm


