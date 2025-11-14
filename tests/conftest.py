"""
conftest.py
───────────────────────────────────────────────
Shared pytest fixtures and utilities for testing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta


@pytest.fixture
def sample_raw_data():
    """Create a sample raw dataset for testing preprocessing."""
    np.random.seed(42)
    n_samples = 100

    # Create date range
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    data = {
        'date': dates,
        'Usage_kWh': np.random.uniform(10, 150, n_samples),
        'Lagging_Current_Reactive.Power_kVarh': np.random.uniform(0, 90, n_samples),
        'Leading_Current_Reactive_Power_kVarh': np.random.uniform(0, 25, n_samples),
        'CO2(tCO2)': np.random.uniform(0, 0.015, n_samples),
        'Lagging_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'Leading_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'NSM': np.random.randint(1000, 80000, n_samples),
        'WeekStatus': np.random.choice(['Weekday', 'Weekend'], n_samples),
        'Day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                         'Friday', 'Saturday', 'Sunday'], n_samples),
        'Load_Type': np.random.choice(['Light_Load', 'Medium_Load', 'Maximum_Load'], n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_data_with_issues():
    """Create a sample dataset with data quality issues for testing cleaning."""
    np.random.seed(42)
    n_samples = 50

    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    data = {
        'date': dates,
        'Usage_kWh': np.random.uniform(10, 150, n_samples),
        'Lagging_Current_Reactive.Power_kVarh': np.random.uniform(0, 90, n_samples),
        'Leading_Current_Reactive_Power_kVarh': np.random.uniform(0, 25, n_samples),
        'CO2(tCO2)': np.random.uniform(0, 0.015, n_samples),
        'Lagging_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'Leading_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'NSM': np.random.randint(1000, 80000, n_samples),
        'WeekStatus': np.random.choice(['Weekday', 'Weekend'], n_samples),
        'Day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday'], n_samples),
        'Load_Type': np.random.choice(['Light_Load', 'Medium_Load', 'Maximum_Load'], n_samples)
    }

    df = pd.DataFrame(data)

    # Introduce issues
    # Missing values
    df.loc[0:5, 'Usage_kWh'] = np.nan
    df.loc[10:15, 'WeekStatus'] = None

    # Out-of-range values
    df.loc[20:25, 'Lagging_Current_Power_Factor'] = np.random.uniform(100, 200, 6)
    df.loc[30:35, 'CO2(tCO2)'] = np.random.uniform(20, 50, 6)
    df.loc[40:45, 'NSM'] = np.random.randint(90000, 200000, 6)

    return df


@pytest.fixture
def sample_processed_data():
    """Create a sample processed dataset for testing feature engineering."""
    np.random.seed(42)
    n_samples = 100

    start_date = datetime(2018, 1, 1, 0, 0, 0)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    data = {
        'date': dates,
        'Usage_kWh': np.random.uniform(10, 150, n_samples),
        'Lagging_Current_Reactive.Power_kVarh': np.random.uniform(0, 90, n_samples),
        'Leading_Current_Reactive_Power_kVarh': np.random.uniform(0, 25, n_samples),
        'CO2(tCO2)': np.random.uniform(0, 0.015, n_samples),
        'Lagging_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'Leading_Current_Power_Factor': np.random.uniform(0.5, 1.0, n_samples),
        'NSM': np.random.randint(1000, 80000, n_samples),
        'WeekStatus': np.random.choice(['WEEKDAY', 'WEEKEND'], n_samples),
        'Day_of_week': np.random.choice(['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
                                         'FRIDAY', 'SATURDAY', 'SUNDAY'], n_samples),
        'Load_Type': np.random.choice(['LIGHT_LOAD', 'MEDIUM_LOAD', 'MAXIMUM_LOAD'], n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features_and_target(sample_processed_data):
    """Create sample feature matrix and target for model testing."""
    df = sample_processed_data.copy()

    # Add some engineered features
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day_of_week_num'] = df['date'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Drop date and target
    y = df['Usage_kWh'].copy()
    X = df.drop(['date', 'Usage_kWh'], axis=1)

    # Convert categoricals to numeric for simplicity
    for col in ['WeekStatus', 'Day_of_week', 'Load_Type']:
        X[col] = pd.Categorical(X[col]).codes

    return X, y


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data files."""
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories
    raw_dir = Path(temp_dir) / "raw"
    processed_dir = Path(temp_dir) / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    yield {
        'root': Path(temp_dir),
        'raw': raw_dir,
        'processed': processed_dir
    }

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    model_dir = Path(temp_dir) / "models"
    model_dir.mkdir(parents=True)

    yield model_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_mlflow_run(mocker):
    """Mock MLflow run context for testing without actual MLflow tracking."""
    mock_run = mocker.MagicMock()
    mock_run.__enter__ = mocker.MagicMock(return_value=mock_run)
    mock_run.__exit__ = mocker.MagicMock(return_value=False)
    mock_run.info.run_id = "test_run_id_12345"

    mocker.patch('mlflow.start_run', return_value=mock_run)
    mocker.patch('mlflow.log_param')
    mocker.patch('mlflow.log_metric')
    mocker.patch('mlflow.log_artifact')
    mocker.patch('mlflow.sklearn.log_model')

    return mock_run


@pytest.fixture(autouse=True)
def silence_warnings():
    """Automatically silence warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore")
    yield
    warnings.filterwarnings("default")


def assert_dataframe_equal(df1, df2, **kwargs):
    """Helper function to compare DataFrames with better error messages."""
    pd.testing.assert_frame_equal(df1, df2, **kwargs)


def assert_no_missing_values(df):
    """Helper to assert DataFrame has no missing values."""
    assert df.isnull().sum().sum() == 0, f"DataFrame has {df.isnull().sum().sum()} missing values"


def assert_column_types(df, expected_types):
    """Helper to assert DataFrame columns have expected types."""
    for col, expected_type in expected_types.items():
        actual_type = df[col].dtype
        assert actual_type == expected_type, \
            f"Column '{col}' has type {actual_type}, expected {expected_type}"
