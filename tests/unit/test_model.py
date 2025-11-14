"""
test_model.py
───────────────────────────────────────────────
Unit tests for model training and prediction
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from imodels import RuleFitRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from models.rulefit_trainer import RuleFitTrainer
from models.model_prediction import RuleFitPredictor


@pytest.mark.unit
class TestRuleFitTrainer:
    """Tests for RuleFitTrainer class."""

    def test_initialization(self, temp_data_dir, temp_model_dir):
        """Test RuleFitTrainer initialization."""
        trainer = RuleFitTrainer(
            data_dir=str(temp_data_dir['processed']),
            model_dir=str(temp_model_dir)
        )

        assert trainer.data_dir == temp_data_dir['processed']
        assert trainer.model_dir == temp_model_dir
        assert trainer.model_dir.exists()

    def test_train_rulefit(self, sample_features_and_target, mock_mlflow_run):
        """Test training RuleFit model."""
        X, y = sample_features_and_target

        trainer = RuleFitTrainer()
        model = trainer.train_rulefit(X, y, max_rules=10, tree_size=3)

        # Assertions
        assert model is not None
        assert isinstance(model, RuleFitRegressor)
        assert hasattr(model, 'predict')

    def test_model_predictions_shape(self, sample_features_and_target, mock_mlflow_run):
        """Test that model predictions have correct shape."""
        X, y = sample_features_and_target

        trainer = RuleFitTrainer()
        model = trainer.train_rulefit(X, y, max_rules=10)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


@pytest.mark.unit
class TestRuleFitPredictor:
    """Tests for RuleFitPredictor class."""

    def test_initialization(self, temp_model_dir, temp_data_dir):
        """Test RuleFitPredictor initialization."""
        model_path = temp_model_dir / 'rulefit.pkl'
        preprocessor_path = temp_data_dir['processed'] / 'preprocessor.pkl'

        predictor = RuleFitPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path)
        )

        assert predictor.model_path == model_path
        assert predictor.preprocessor_path == preprocessor_path

    def test_load_model(self, temp_model_dir, sample_features_and_target):
        """Test loading a saved model."""
        X, y = sample_features_and_target

        # Train and save a model
        model = RuleFitRegressor(max_rules=10, random_state=42)
        model.fit(X, y)
        model_path = temp_model_dir / 'test_model.pkl'
        joblib.dump(model, model_path)

        # Load model
        predictor = RuleFitPredictor(
            model_path=str(model_path),
            preprocessor_path='dummy_path.pkl'
        )
        predictor.load_model()

        # Assertions
        assert predictor.model is not None
        assert isinstance(predictor.model, RuleFitRegressor)
