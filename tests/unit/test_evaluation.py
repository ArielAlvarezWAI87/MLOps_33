"""
test_evaluation.py
───────────────────────────────────────────────
Unit tests for model evaluation and data drift simulation.
"""

from pathlib import Path
import sys

import pytest

from src.evaluation.evaluate import (
    evaluate_current_model,
    evaluate_model_under_drift,
)


@pytest.mark.unit
class TestModelEvaluation:
    """Tests for baseline model evaluation."""

    def test_evaluate_current_model_returns_metrics(self):
        """evaluate_current_model debe regresar un dict con mae, rmse y r2."""
        metrics = evaluate_current_model(log_to_mlflow=False)

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

    def test_evaluate_model_under_drift_structure(self):
        """
        La función debe regresar un dict con las secciones esperadas:
        baseline, drift, delta y alert.
        """
        results = evaluate_model_under_drift(
            log_to_mlflow=False, shift_fraction=0.2
        )

        assert isinstance(results, dict)
        assert "baseline" in results
        assert "drift" in results
        assert "delta" in results
        assert "alert" in results

        for section in ["baseline", "drift"]:
            assert isinstance(results[section], dict)
            assert "mae" in results[section]
            assert "rmse" in results[section]
            assert "r2" in results[section]

        assert "r2_drop" in results["delta"]
        assert "rmse_increase" in results["delta"]
        assert "mae_increase" in results["delta"]

        assert "drift_detected" in results["alert"]
        assert "r2_drop_alert_threshold" in results["alert"]

    def test_evaluate_model_under_drift_detects_degradation(self):
        """
        Con un shift razonable, esperamos que el desempeño bajo drift
        sea peor que el baseline (r2_drop > 0).
        """
        results = evaluate_model_under_drift(
            log_to_mlflow=False, shift_fraction=0.2
        )

        r2_drop = results["delta"]["r2_drop"]
        assert r2_drop > 0.0
