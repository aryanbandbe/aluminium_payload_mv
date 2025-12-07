import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# === CONFIG ===
DATASET_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PROCESSES = {
    "al_primary_prebake": "al_primary_prebake_dataset.csv",
    "al_primary_soderberg": "al_primary_soderberg_dataset.csv",
    "al_secondary_remelt": "al_secondary_remelt_dataset.csv"
}

print("\n=== 🚀 Training Combined Imputation Models for Aluminium Processes ===\n")


def train_combined_model(process_name):
    print(f"\n🧩 Training for process: {process_name}")
    dataset_path = os.path.join(DATASET_DIR, PROCESSES[process_name])
    df = pd.read_csv(dataset_path)

    # Drop non-numeric / identifiers
    df = df.select_dtypes(include=[np.number])

    # Remove rows that are all NaN
    df.dropna(how="all", inplace=True)

    if df.empty:
        print(f"⚠️ Skipping {process_name} — no data.")
        return

    imputed_models = {}

    # Train one XGBoost regressor per column to fill its missing values
    for target_col in df.columns:
        if df[target_col].isnull().sum() == 0:
            continue

        print(f"🔧 Training imputer for '{target_col}' ...")

        df_train = df[df[target_col].notnull()]
        df_pred = df[df[target_col].isnull()]

        if len(df_train) < 5:
            print(f"⚠️ Skipping '{target_col}' — insufficient data.")
            continue

        X = df_train.drop(columns=[target_col])

        # ✅ Drop features that are entirely NaN
        null_cols = [c for c in X.columns if X[c].isnull().all()]
        if null_cols:
            print(f"⚠️ Dropping columns with all missing values: {null_cols}")
            X = X.drop(columns=null_cols)

        # ✅ Safely impute remaining features
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)

        y = df_train[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"✅ {target_col}: RMSE = {rmse:.4f}")

        imputed_models[target_col] = {
            "model": model,
            "used_features": list(X.columns)
        }

    # ✅ Save combined model
    model_path = os.path.join(MODEL_DIR, f"xgb_imputer_{process_name}.joblib")
    joblib.dump(imputed_models, model_path)
    print(f"💾 Saved combined model for {process_name} → {model_path}")


for process in PROCESSES:
    train_combined_model(process)

print("\n🎉 All process models trained and saved successfully!")
