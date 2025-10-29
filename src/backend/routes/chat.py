from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.llm import get_llm


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=str)
def chat(request: str):
    text = (request.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must be a non-empty string")

    try:
        llm = get_llm()
        resp = llm.invoke(text)
        content = getattr(resp, "content", "")
        if not content:
            raise ValueError("Empty response from LLM")
        return ChatResponse(output=content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


