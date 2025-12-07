from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import json
import os
import requests
import pandas as pd

# === CONFIG ===
MODEL_DIR = "models"
BRIGHTWAY_URL = "https://brightway-engine.onrender.com/aluminium/run"

# === Load all models ===
MODELS = {}
for file in os.listdir(MODEL_DIR):
    if file.startswith("xgb_imputer_") and file.endswith(".joblib"):
        process_name = file.replace("xgb_imputer_", "").replace(".joblib", "")
        MODELS[process_name] = joblib.load(os.path.join(MODEL_DIR, file))

print(f"✅ Loaded models for processes: {list(MODELS.keys())}\n")

# === Initialize FastAPI app ===
app = FastAPI(
    title="Aluminium Imputation API",
    description="Fills missing values in aluminium process payloads and sends results to Brightway Engine.",
    version="1.0.0"
)

# Allow CORS (so Firebase/Render can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/impute")
async def impute_missing_values(payload: dict):
    """Impute missing values for aluminium process payloads and forward to Brightway Engine."""

    try:
        metal = payload.get("metal")
        route = payload.get("route")
        inputs = payload.get("inputs", {})

        if metal != "aluminium":
            raise HTTPException(status_code=400, detail="❌ Only aluminium supported.")
        if route not in MODELS:
            raise HTTPException(status_code=400, detail=f"❌ Unknown route '{route}'.")

        model_bundle = MODELS[route]
        filled_inputs = {}

        # Loop through each stage (mining, extraction, manufacturing)
        for stage, params in inputs.items():
            filled_inputs[stage] = {}
            for key, value in params.items():
                # If missing/null/zero — fill it if model exists
                if value in [None, "", 0, "0"] and key in model_bundle:
                    bundle = model_bundle[key]
                    model = bundle["model"]
                    used_features = bundle["used_features"]

                    X_pred = np.zeros((1, len(used_features)))
                    pred_value = round(float(model.predict(X_pred)[0]), 4)
                    filled_inputs[stage][key] = pred_value
                    print(f"🧩 ({stage}) Filled '{key}' → {pred_value}")
                else:
                    filled_inputs[stage][key] = value

        payload["inputs"] = filled_inputs

        print("\n✅ Final Filled Payload:\n", json.dumps(payload, indent=2))

        # === Send the imputed payload to Brightway Engine ===
        response = requests.post(BRIGHTWAY_URL, json=payload)

        if response.status_code == 200:
            print("🌍 Brightway Engine Response: SUCCESS")
            return {
                "status": "success",
                "message": "Payload imputed and processed successfully.",
                "brightway_response": response.json(),
                "final_payload": payload
            }
        else:
            print("⚠️ Brightway Engine Error:", response.text)
            raise HTTPException(status_code=500, detail=f"Brightway Engine Error: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "✅ Aluminium Imputation API is running successfully!"}
