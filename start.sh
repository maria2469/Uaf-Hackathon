#!/bin/sh
# ===============================================
# MEDI-AI FastAPI backend startup for Railpack
# ===============================================

echo "🚀 Starting MEDI-AI Backend..."

# Ensure pip exists
python3 -m ensurepip

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
python3 -m pip install -r backend/requirements.txt

# Start FastAPI backend
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080
