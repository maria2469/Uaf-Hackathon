#!/bin/bash
# upgrade pip and install requirements
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# start FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000
