from fastapi import FastAPI
from .routes.chat import router as chat_router


app = FastAPI(title="Healthcare Agent API", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(chat_router, prefix="")


