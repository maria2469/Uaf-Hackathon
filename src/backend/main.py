from fastapi import FastAPI, HTTPException
from .coordinator_service import run_patient_workflow

app = FastAPI(title="Healthcare AI Backend API")

@app.get("/patient/{patient_id}/summary")
def get_summary(patient_id: str):
    try:
        return run_patient_workflow(patient_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
