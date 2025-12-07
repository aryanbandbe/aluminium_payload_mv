#!/bin/bash
set -e

echo "🚀 Setting up Aluminium Imputation API on Render..."

# Create a models directory
mkdir -p models

# Download latest trained models from GitHub release
echo "⬇️ Downloading model files from GitHub release..."
wget -O models/xgb_imputer_al_primary_prebake.joblib https://github.com/aryanbandbe/aluminium_payload_mv/releases/download/v1.0.0/xgb_imputer_al_primary_prebake.joblib
wget -O models/xgb_imputer_al_primary_soderberg.joblib https://github.com/aryanbandbe/aluminium_payload_mv/releases/download/v1.0.0/xgb_imputer_al_primary_soderberg.joblib
wget -O models/xgb_imputer_al_secondary_remelt.joblib https://github.com/aryanbandbe/aluminium_payload_mv/releases/download/v1.0.0/xgb_imputer_al_secondary_remelt.joblib

echo "✅ Model files downloaded successfully."

# Start the FastAPI app
echo "🚀 Launching FastAPI app..."
exec uvicorn main_api:app --host 0.0.0.0 --port $PORT
