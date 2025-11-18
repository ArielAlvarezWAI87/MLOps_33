"""
test_evaluation.py
───────────────────────────────────────────────
Unit tests for model evaluation and data drift simulation.
"""

from pathlib import Path
import sys

import pytest

from src.evaluation.evaluate import (
    evaluate_model,
    evaluate_model_with_drift,
)


@pytest.mark.unit
class TestModelEvaluation:
    """Tests for baseline model evaluation."""

    def test_evaluate_model_returns_metrics(self):
        """evaluate_model debe regresar un dict con mae, rmse y r2."""
        processed_csv_path = "data/processed/steel_energy_processed.csv"
        model_path = "models/rulefit.pkl"

        metrics = evaluate_model(
            processed_csv_path=processed_csv_path,
            model_path=model_path,
            log_to_mlflow=False
        )

        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

        # Valores razonables (no NaNs, no negativos extremos)
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= 0.0
        # En regresión R2 puede ser negativo, pero aquí esperamos un modelo decente
        assert metrics["r2"] <= 1.0


@pytest.mark.unit
class TestModelDriftEvaluation:
    """Tests for model evaluation under simulated data drift."""

    def test_evaluate_model_with_drift_returns_metrics(self):
        """
        evaluate_model_with_drift debe regresar un dict con mae, rmse y r2
        para las predicciones bajo drift.
        """
        processed_csv_path = "data/processed/steel_energy_processed.csv"
        model_path = "models/rulefit.pkl"

        drift_metrics = evaluate_model_with_drift(
            processed_csv_path=processed_csv_path,
            model_path=model_path,
            shift_fraction=0.2,
            log_to_mlflow=False
        )

        assert isinstance(drift_metrics, dict)
        assert "mae" in drift_metrics
        assert "rmse" in drift_metrics
        assert "r2" in drift_metrics

        # Valores razonables
        assert drift_metrics["mae"] >= 0.0
        assert drift_metrics["rmse"] >= 0.0
        assert drift_metrics["r2"] <= 1.0

    def test_evaluate_model_with_drift_detects_degradation(self):
        """
        Con un shift razonable, esperamos que el desempeño bajo drift
        sea peor que el baseline (mayor RMSE, menor R2).
        """
        processed_csv_path = "data/processed/steel_energy_processed.csv"
        model_path = "models/rulefit.pkl"

        # Obtener métricas baseline
        baseline_metrics = evaluate_model(
            processed_csv_path=processed_csv_path,
            model_path=model_path,
            log_to_mlflow=False
        )

        # Obtener métricas bajo drift
        drift_metrics = evaluate_model_with_drift(
            processed_csv_path=processed_csv_path,
            model_path=model_path,
            shift_fraction=0.25,
            log_to_mlflow=False
        )

        # Verificar degradación: RMSE aumenta y R2 disminuye
        assert drift_metrics["rmse"] > baseline_metrics["rmse"]
        assert drift_metrics["r2"] < baseline_metrics["r2"]
