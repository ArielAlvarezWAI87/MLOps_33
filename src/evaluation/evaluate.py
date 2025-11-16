"""
Módulo de evaluación del modelo RuleFit para
consumo de energía en acero.

Responsabilidades:
- Cargar datos procesados (X_features, y_target).
- Cargar el modelo entrenado.
- Calcular métricas de regresión (MAE, RMSE, R2).
- Opcionalmente registrar resultados en MLflow.
- Simular data drift y evaluar el impacto en el desempeño.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import load_processed_data
from config.mlflow_config import MLflowConfig
import mlflow


def load_model(model_path: str = "models/rulefit.pkl"):
    """
    Carga el modelo entrenado desde disco.

    Parameters
    ----------
    model_path : str
        Ruta relativa al archivo .pkl del modelo entrenado.

    Returns
    -------
    model :
        Modelo ya cargado listo para hacer predicciones.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    return model


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de regresión estándar.

    Returns
    -------
    metrics : dict
        Diccionario con MAE, RMSE y R2.
    """
    mae = mean_absolute_error(y_true, y_pred)

    # Algunas versiones de scikit-learn no soportan el argumento 'squared'.
    # Calculamos primero el MSE y luego sacamos la raíz cuadrada para obtener el RMSE.
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def simulate_mean_shift_drift(
    X: pd.DataFrame,
    shift_fraction: float = 0.2,
    numeric_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Simula data drift aplicando un desplazamiento (shift) en las variables numéricas.

    Parameters
    ----------
    X : pd.DataFrame
        Features originales (datos de referencia).
    shift_fraction : float
        Fracción de la desviación estándar que se suma a cada columna numérica.
        Por ejemplo, 0.2 significa que se suma 0.2 * std(columna).
    numeric_columns : list, optional
        Lista de columnas numéricas a las que se aplicará el shift.
        Si es None, se usan todas las columnas numéricas de X.

    Returns
    -------
    X_drifted : pd.DataFrame
        Copia de X con data drift simulado.
    """
    X_drifted = X.copy()

    if numeric_columns is None:
        numeric_columns = X_drifted.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_columns:
        std = X_drifted[col].std()
        shift = shift_fraction * std
        X_drifted[col] = X_drifted[col] + shift

    return X_drifted


def evaluate_current_model(
    data_dir: str = "data/processed",
    model_path: str = "models/rulefit.pkl",
    log_to_mlflow: bool = True,
    run_name: Optional[str] = "offline_evaluation"
) -> Dict[str, float]:
    """
    Evalúa el modelo actual usando los datos procesados completos.

    Flujo:
    - Carga X_features, y_target con load_processed_data.
    - Carga el modelo rulefit.pkl.
    - Calcula métricas.
    - (Opcional) Registra métricas en MLflow.
    """
    # 1) Cargar datos procesados
    X, y = load_processed_data(data_dir=data_dir)

    # 2) Cargar modelo
    model = load_model(model_path=model_path)

    # 3) Predicciones
    y_pred = model.predict(X)

    # 4) Métricas
    metrics = compute_regression_metrics(y_true=y, y_pred=y_pred)

    # 5) Logging en MLflow (opcional)
    if log_to_mlflow:
        mlflow_cfg = MLflowConfig()
        mlflow_cfg.setup()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("evaluation_data_dir", data_dir)
            mlflow.log_param("evaluation_model_path", model_path)

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

    return metrics


def evaluate_model_under_drift(
    data_dir: str = "data/processed",
    model_path: str = "models/rulefit.pkl",
    shift_fraction: float = 0.2,
    r2_drop_alert_threshold: float = 0.02,
    log_to_mlflow: bool = True,
    run_name: Optional[str] = "drift_evaluation_mean_shift",
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa el impacto de un data drift simulado sobre el desempeño del modelo.

    Flujo:
    - Carga X_features, y_target (datos de referencia).
    - Calcula métricas baseline.
    - Genera un dataset de monitoreo con shift en variables numéricas.
    - Calcula métricas con drift.
    - Compara métricas y define si se activa una alerta.
    """
    # 1) Cargar datos procesados
    X_ref, y_ref = load_processed_data(data_dir=data_dir)

    # 2) Cargar modelo
    model = load_model(model_path=model_path)

    # 3) Baseline (sin drift)
    y_pred_ref = model.predict(X_ref)
    baseline_metrics = compute_regression_metrics(y_true=y_ref, y_pred=y_pred_ref)

    # 4) Generar dataset de monitoreo con drift
    X_drifted = simulate_mean_shift_drift(X_ref, shift_fraction=shift_fraction)
    y_pred_drift = model.predict(X_drifted)
    drift_metrics = compute_regression_metrics(y_true=y_ref, y_pred=y_pred_drift)

    # 5) Comparar métricas
    r2_drop = baseline_metrics["r2"] - drift_metrics["r2"]
    rmse_increase = drift_metrics["rmse"] - baseline_metrics["rmse"]
    mae_increase = drift_metrics["mae"] - baseline_metrics["mae"]

    drift_alert = r2_drop >= r2_drop_alert_threshold

    results = {
        "baseline": baseline_metrics,
        "drift": drift_metrics,
        "delta": {
            "r2_drop": float(r2_drop),
            "rmse_increase": float(rmse_increase),
            "mae_increase": float(mae_increase),
        },
        "alert": {
            "drift_detected": bool(drift_alert),
            "r2_drop_alert_threshold": float(r2_drop_alert_threshold),
        },
    }

    # 6) Guardar dataset de monitoreo (útil para la tarea)
    monitoring_dir = Path("data/monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    X_drifted.to_csv(monitoring_dir / "X_monitor_drifted.csv", index=False)
    pd.DataFrame({"target": y_ref}).to_csv(
        monitoring_dir / "y_monitor_drifted.csv", index=False
    )

    # 7) Logging en MLflow (opcional)
    if log_to_mlflow:
        mlflow_cfg = MLflowConfig()
        mlflow_cfg.setup()

        with mlflow.start_run(run_name=run_name):
            # Parámetros del experimento de drift
            mlflow.log_param("evaluation_data_dir", data_dir)
            mlflow.log_param("evaluation_model_path", model_path)
            mlflow.log_param("shift_fraction", shift_fraction)
            mlflow.log_param("r2_drop_alert_threshold", r2_drop_alert_threshold)

            # Métricas baseline
            for name, value in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{name}", value)

            # Métricas con drift
            for name, value in drift_metrics.items():
                mlflow.log_metric(f"drift_{name}", value)

            # Deltas
            mlflow.log_metric("r2_drop", r2_drop)
            mlflow.log_metric("rmse_increase", rmse_increase)
            mlflow.log_metric("mae_increase", mae_increase)

            # Alerta (0 o 1)
            mlflow.log_metric("drift_alert", 1.0 if drift_alert else 0.0)

    return results


if __name__ == "__main__":
    # Evaluación baseline
    metrics = evaluate_current_model()
    print("Evaluation metrics (baseline):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Evaluación con drift simulado
    print("\nEvaluating model under simulated data drift...")
    drift_results = evaluate_model_under_drift()
    print("\nBaseline metrics:")
    for k, v in drift_results["baseline"].items():
        print(f"  {k}: {v:.4f}")

    print("\nDrifted metrics:")
    for k, v in drift_results["drift"].items():
        print(f"  {k}: {v:.4f}")

    print("\nDelta:")
    for k, v in drift_results["delta"].items():
        print(f"  {k}: {v:.4f}")

    print("\nDrift alert:", drift_results["alert"])