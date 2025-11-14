"""
test_features.py
───────────────────────────────────────────────
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from features.feature_engineering import FeatureEngineer


@pytest.mark.unit
class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_initialization(self, temp_data_dir):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(
            raw_processed_dir=str(temp_data_dir['processed']),
            save_dir=str(temp_data_dir['processed'])
        )

        assert engineer.raw_processed_dir == temp_data_dir['processed']
        assert engineer.save_dir == temp_data_dir['processed']
        assert engineer.save_dir.exists()

    def test_engineer_features_temporal(self, sample_processed_data):
        """Test temporal feature engineering."""
        engineer = FeatureEngineer()
        df_eng = engineer.engineer_features(sample_processed_data.copy())

        # Check temporal features exist
        temporal_features = ['year', 'month', 'day', 'hour', 'day_of_week_num', 'quarter', 'is_weekend']
        for feature in temporal_features:
            assert feature in df_eng.columns, f"{feature} should be in engineered data"

        # Check ranges
        assert df_eng['month'].between(1, 12).all(), "Month should be 1-12"
        assert df_eng['day'].between(1, 31).all(), "Day should be 1-31"
        assert df_eng['hour'].between(0, 23).all(), "Hour should be 0-23"

    def test_engineer_features_cyclical(self, sample_processed_data):
        """Test cyclical feature engineering."""
        engineer = FeatureEngineer()
        df_eng = engineer.engineer_features(sample_processed_data.copy())

        # Check cyclical features exist
        cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
        for feature in cyclical_features:
            assert feature in df_eng.columns, f"{feature} should be in engineered data"

        # Check that sine/cosine values are in [-1, 1]
        for feature in cyclical_features:
            assert df_eng[feature].between(-1, 1).all(), \
                f"{feature} should be between -1 and 1"
