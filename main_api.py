from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import json
import os

# === CONFIG ===
MODEL_DIR = "models"

# === Load all models ===
MODELS = {}
if os.path.exists(MODEL_DIR):
    for file in os.listdir(MODEL_DIR):
        if file.startswith("xgb_imputer_") and file.endswith(".joblib"):
            process_name = file.replace("xgb_imputer_", "").replace(".joblib", "")
            MODELS[process_name] = joblib.load(os.path.join(MODEL_DIR, file))

print(f"✅ Loaded models for routes: {list(MODELS.keys())}\n")

# === Initialize FastAPI app ===
app = FastAPI(
    title="Aluminium Imputation API",
    description="Fills missing values in aluminium process payloads and returns filled payload to the caller (Firebase).",
    version="2.0.0"
)

# Allow CORS (so Firebase/Render can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can later restrict this to your Firebase domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/impute/{metal}")
async def impute_missing_values(metal: str, payload: dict):
    """
    Impute missing values for aluminium process payloads and return the filled payload
    directly to the requesting client (Firebase).
    """

    try:
        route = payload.get("route")
        inputs = payload.get("inputs", {})

        if metal.lower() != "aluminium":
            raise HTTPException(status_code=400, detail="❌ Only aluminium supported for this API.")
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

                    # Create dummy feature vector for prediction
                    X_pred = np.zeros((1, len(used_features)))
                    pred_value = round(float(model.predict(X_pred)[0]), 4)
                    filled_inputs[stage][key] = pred_value
                    print(f"🧩 ({stage}) Filled '{key}' → {pred_value}")
                else:
                    filled_inputs[stage][key] = value

        payload["inputs"] = filled_inputs

        print("\n✅ Final Filled Payload (sending back to Firebase):\n", json.dumps(payload, indent=2))

        # Return the filled payload directly to Firebase
        return {
            "status": "success",
            "message": "Payload imputed successfully.",
            "filled_payload": payload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "✅ Aluminium Imputation API is running successfully and ready to fill missing values!"}
