"""
test_pipeline.py
───────────────────────────────────────────────
Integration tests for the full MLOps pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from imodels import RuleFitRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from data.preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.rulefit_trainer import RuleFitTrainer
from models.model_prediction import RuleFitPredictor


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete data -> model -> prediction pipeline."""

    def test_preprocessing_to_feature_engineering(self, temp_data_dir, sample_raw_data):
        """Test pipeline from raw data to engineered features."""
        # Step 1: Preprocessing
        raw_csv = temp_data_dir['raw'] / 'steel_energy_original.csv'
        sample_raw_data.to_csv(raw_csv, index=False)

        preprocessor = DataPreprocessor(
            raw_dir=str(temp_data_dir['raw']),
            processed_dir=str(temp_data_dir['processed'])
        )
        df_clean = preprocessor.run()

        # Step 2: Feature Engineering
        engineer = FeatureEngineer(
            raw_processed_dir=str(temp_data_dir['processed']),
            save_dir=str(temp_data_dir['processed'])
        )
        X_transformed, y, preprocessor_pipeline = engineer.run()

        # Assertions
        assert df_clean is not None
        assert X_transformed is not None
        assert y is not None
        assert len(X_transformed) == len(y)

    def test_feature_engineering_to_model_training(self, temp_data_dir, temp_model_dir,
                                                   sample_raw_data, mock_mlflow_run):
        """Test pipeline from raw data to trained model."""
        # Step 1: Preprocessing
        raw_csv = temp_data_dir['raw'] / 'steel_energy_original.csv'
        sample_raw_data.to_csv(raw_csv, index=False)

        preprocessor = DataPreprocessor(
            raw_dir=str(temp_data_dir['raw']),
            processed_dir=str(temp_data_dir['processed'])
        )
        preprocessor.run()

        # Step 2: Feature Engineering
        engineer = FeatureEngineer(
            raw_processed_dir=str(temp_data_dir['processed']),
            save_dir=str(temp_data_dir['processed'])
        )
        engineer.run()

        # Step 3: Model Training
        trainer = RuleFitTrainer(
            data_dir=str(temp_data_dir['processed']),
            model_dir=str(temp_model_dir)
        )
        model = trainer.run()

        # Assertions
        assert model is not None
        assert isinstance(model, RuleFitRegressor)
