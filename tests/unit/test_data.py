"""
test_data.py
───────────────────────────────────────────────
Unit tests for data loading and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from data.load_data import load_raw_data, load_processed_data
from data.preprocessing import DataPreprocessor


# =============================================================================
# Tests for load_data.py
# =============================================================================

@pytest.mark.unit
class TestLoadData:
    """Tests for data loading functions."""

    def test_load_raw_data_success(self, temp_data_dir, sample_raw_data):
        """Test loading raw data from CSV file."""
        # Save sample data to temp file
        csv_path = temp_data_dir['raw'] / 'test_data.csv'
        sample_raw_data.to_csv(csv_path, index=False)

        # Load the data
        df = load_raw_data(str(csv_path))

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_data)
        assert list(df.columns) == list(sample_raw_data.columns)

    def test_load_raw_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data('nonexistent_file.csv')

    def test_load_processed_data_success(self, temp_data_dir, sample_features_and_target):
        """Test loading processed features and target."""
        X, y = sample_features_and_target

        # Save to temp directory
        X.to_csv(temp_data_dir['processed'] / 'X_features.csv', index=False)
        pd.DataFrame(y).to_csv(temp_data_dir['processed'] / 'y_target.csv', index=False)

        # Load back
        X_loaded, y_loaded = load_processed_data(str(temp_data_dir['processed']))

        # Assertions
        assert isinstance(X_loaded, pd.DataFrame)
        assert isinstance(y_loaded, np.ndarray)
        assert X_loaded.shape == X.shape
        assert y_loaded.shape == y.shape

    def test_load_processed_data_missing_files(self, temp_data_dir):
        """Test that error is raised when processed files are missing."""
        with pytest.raises(FileNotFoundError):
            load_processed_data(str(temp_data_dir['processed']))


# =============================================================================
# Tests for preprocessing.py - DataPreprocessor class
# =============================================================================

@pytest.mark.unit
class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_initialization(self, temp_data_dir):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(
            raw_dir=str(temp_data_dir['raw']),
            processed_dir=str(temp_data_dir['processed'])
        )

        assert preprocessor.raw_dir == temp_data_dir['raw']
        assert preprocessor.processed_dir == temp_data_dir['processed']
        assert preprocessor.processed_dir.exists()

    def test_load_data(self, temp_data_dir, sample_raw_data):
        """Test loading data from raw directory."""
        # Save sample data
        csv_path = temp_data_dir['raw'] / 'steel_energy_original.csv'
        sample_raw_data.to_csv(csv_path, index=False)

        # Initialize preprocessor and load data
        preprocessor = DataPreprocessor(
            raw_dir=str(temp_data_dir['raw']),
            processed_dir=str(temp_data_dir['processed'])
        )
        df = preprocessor.load_data()

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_data)

    def test_clean_data_type_conversions(self, sample_raw_data_with_issues):
        """Test that numeric and datetime conversions work correctly."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_raw_data_with_issues.copy())

        # Check numeric columns are numeric type
        numeric_columns = [
            'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
            'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM'
        ]
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df_clean[col]), \
                f"Column {col} should be numeric"

        # Check date is datetime
        assert pd.api.types.is_datetime64_any_dtype(df_clean['date']), \
            "date column should be datetime"

    def test_clean_data_categorical_handling(self, sample_raw_data_with_issues):
        """Test that categorical variables are properly cleaned."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_raw_data_with_issues.copy())

        categorical_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']

        for col in categorical_cols:
            # Should be uppercase and stripped
            assert df_clean[col].str.isupper().all(), \
                f"Column {col} should be uppercase"
            # Should have no missing values
            assert df_clean[col].notna().all(), \
                f"Column {col} should have no missing values"

    def test_clean_data_no_missing_values_final(self, sample_raw_data_with_issues):
        """Test that cleaned data has no missing values."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_raw_data_with_issues)

        # Should have no missing values
        assert df_clean.isnull().sum().sum() == 0, \
            "Cleaned data should have no missing values"
