"""
evaluate.py — SELF-SUFFICIENT VERSION
───────────────────────────────────────────────────────────────
This evaluator does NOT require:
    data/tmp/X_features.csv
    data/tmp/y_target.csv

It instead loads:
    data/processed/steel_energy_processed.csv

And internally runs:
    FeatureEngineer.run()

So that evaluation ALWAYS uses the correct:
    engineered + scaled features
    target
    preprocessor
matching exactly the training pipeline.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import joblib
import mlflow
import sys
import os


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================
mlflow.set_tracking_uri("file:../mlruns")


# ============================================================
# IMPORT PHASE 2 MODULES
# ============================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.feature_engineering import FeatureEngineer


# ============================================================
# LOAD CLEANED DATAFRAME
# ============================================================

def load_clean_dataframe(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_rulefit_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


# ============================================================
# METRICS
# ============================================================

def compute_regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ============================================================
# DRIFT SIMULATION ON ENGINEERED FEATURES
# ============================================================

def simulate_drift(X: pd.DataFrame, shift_fraction=0.2):
    Xd = X.copy()
    numeric_cols = Xd.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Xd[col] += shift_fraction * Xd[col].std()

    return Xd


# ============================================================
# BASELINE EVALUATION
# ============================================================

def evaluate_model(
    processed_csv_path: str,
    model_path: str,
    log_to_mlflow=False,
    experiment="baseline_evaluation",
    run="baseline"
):
    """
    Full end-to-end evaluation (baseline):
    processed CSV → FeatureEngineering → predict → metrics
    """

    # --------------------------------------------------------
    # 1) LOAD CLEANED DATA
    # --------------------------------------------------------
    df_clean = load_clean_dataframe(processed_csv_path)

    # --------------------------------------------------------
    # 2) REBUILD FEATURES
    # --------------------------------------------------------
    fe = FeatureEngineer(
        raw_processed_dir="data/processed",
        save_dir="data/tmp"
    )

    X_processed, y, preprocessor = fe.run(
        filename=Path(processed_csv_path).name
    )

    # --------------------------------------------------------
    # 3) LOAD MODEL
    # --------------------------------------------------------
    model = load_rulefit_model(model_path)

    # --------------------------------------------------------
    # 4) PREDICT + METRICS
    # --------------------------------------------------------
    y_pred = model.predict(X_processed)
    metrics = compute_regression_metrics(y, y_pred)

    # --------------------------------------------------------
    # 5) OPTIONAL MLflow logging
    # --------------------------------------------------------
    if log_to_mlflow:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run):

            # Parameters
            mlflow.log_param("processed_csv", processed_csv_path)
            mlflow.log_param("model_path", model_path)

            # Metrics
            for k, v in metrics.items():
                mlflow.log_metric(f"baseline_{k}", v)

            # Artifacts
            mlflow.log_artifact(model_path, artifact_path="model")

            preprocessor_path = "data/tmp/preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

    return metrics


# ============================================================
# DRIFT EVALUATION
# ============================================================

def evaluate_model_with_drift(
    processed_csv_path: str,
    model_path: str,
    shift_fraction=0.25,
    log_to_mlflow=False,
    experiment="drift_eval",
    run="drift_eval"
):
    """
    Evaluate a model under baseline and drifted conditions.
    """

    # --------------------------------------------------------
    # 1) LOAD CLEANED DATA
    # --------------------------------------------------------
    df_clean = load_clean_dataframe(processed_csv_path)

    # --------------------------------------------------------
    # 2) REBUILD FEATURES
    # --------------------------------------------------------
    fe = FeatureEngineer(
        raw_processed_dir="data/processed",
        save_dir="data/tmp"
    )

    X_processed, y, preprocessor = fe.run(
        filename=Path(processed_csv_path).name
    )

    # --------------------------------------------------------
    # 3) LOAD MODEL
    # --------------------------------------------------------
    model = load_rulefit_model(model_path)

    # --------------------------------------------------------
    # 4) BASELINE METRICS
    # --------------------------------------------------------
    y_pred_base = model.predict(X_processed)
    baseline_metrics = compute_regression_metrics(y, y_pred_base)

    # --------------------------------------------------------
    # 5) DRIFTED METRICS
    # --------------------------------------------------------
    X_drift = simulate_drift(X_processed, shift_fraction)
    y_pred_drift = model.predict(X_drift)
    drift_metrics = compute_regression_metrics(y, y_pred_drift)

    # --------------------------------------------------------
    # 6) OPTIONAL MLflow logging
    # --------------------------------------------------------
    if log_to_mlflow:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run):

            # Parameters
            mlflow.log_param("processed_csv", processed_csv_path)
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("shift_fraction", shift_fraction)

            # Metrics
            for k, v in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{k}", v)

            for k, v in drift_metrics.items():
                mlflow.log_metric(f"drift_{k}", v)

            # Artifacts
            mlflow.log_artifact(model_path, artifact_path="model")

            preprocessor_path = "data/tmp/preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

    return {"drift": drift_metrics}

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    PROCESSED_PATH = "data/processed/steel_energy_processed.csv"
    MODEL_PATH = "models/rulefit.pkl"
    experiment = "baseline_drift_evaluation"

    print("📊 BASELINE EVALUATION")
    base = evaluate_model(PROCESSED_PATH, MODEL_PATH, log_to_mlflow=True,experiment=experiment)
    

    print("\n🌪 DRIFT EVALUATION")
    drift = evaluate_model_with_drift(PROCESSED_PATH, MODEL_PATH, log_to_mlflow=True,experiment=experiment)
    
    print("\n--- RESULTS ---")
    print("BASELINE METRICS:", base)
    print("DRIFT METRICS:", drift)
