import json
import joblib
import numpy as np
import os
import pandas as pd

MODEL_DIR = "models"

# === Load all trained models ===
MODELS = {}
for file in os.listdir(MODEL_DIR):
    if file.startswith("xgb_imputer_") and file.endswith(".joblib"):
        process_name = file.replace("xgb_imputer_", "").replace(".joblib", "")
        MODELS[process_name] = joblib.load(os.path.join(MODEL_DIR, file))

print(f"✅ Loaded models for processes: {list(MODELS.keys())}\n")


def impute_missing_values(payload):
    """Fill missing values in payload based on the corresponding process model."""

    metal = payload.get("metal")
    route = payload.get("route")
    inputs = payload.get("inputs", {})

    if metal != "aluminium":
        raise ValueError("❌ Only aluminium supported in this script.")
    if route not in MODELS:
        raise ValueError(f"❌ Unknown route '{route}'.")

    model_bundle = MODELS[route]
    filled_inputs = {}

    for stage, params in inputs.items():
        filled_inputs[stage] = {}

        for key, value in params.items():
            # Check if this parameter needs imputation
            if value in [None, "", 0, "0"] and key in model_bundle:
                bundle = model_bundle[key]
                model = bundle["model"]
                used_features = bundle["used_features"]

                # For prediction, create a simple mean/zero input
                X_pred = np.zeros((1, len(used_features)))
                pred_value = round(float(model.predict(X_pred)[0]), 4)
                filled_inputs[stage][key] = pred_value

                print(f"🧩 ({stage}) Filled '{key}' → {pred_value}")

            else:
                filled_inputs[stage][key] = value

    payload["inputs"] = filled_inputs

    print("\n✅ Final Imputed Payload:\n")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    test_files = [
        "test_payloads/test_prebake.json",
        "test_payloads/test_soderberg.json",
        "test_payloads/test_remelt.json"
    ]

    for test_file in test_files:
        print(f"\n🔍 Testing {test_file} ...")
        with open(test_file) as f:
            payload = json.load(f)
        impute_missing_values(payload)
