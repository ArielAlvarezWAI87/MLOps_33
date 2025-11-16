"""
test_preprocessing.py
────────────────────────────────────────
Unit tests for src.features.preprocessing.DataPreprocessor

"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_raw_df():
    """
    Minimal but representative subset of the raw schema, simulating
    what comes from steel_energy_original.csv before cleaning.
    """
    data = {
        "Usage_kWh": [10, np.nan, 30],
        "Lagging_Current_Reactive.Power_kVarh": [120.0, np.nan, 50.0],
        "Leading_Current_Reactive_Power_kVarh": [10.0, 20.0, 30.0],
        "CO2(tCO2)": [0.01, 0.30, 0.05],  # 0.30 triggers CO2 scaling rule
        "Lagging_Current_Power_Factor": [150.0, 90.0, 80.0],  # 150 triggers scaling
        "Leading_Current_Power_Factor": [110.0, 95.0, 85.0],  # 110 triggers scaling
        "NSM": [90000.0, 100.0, np.nan],  # 90000 triggers NSM scaling / capping
        "WeekStatus": ["weekday", "WEEKEND", None],
        "Day_of_week": ["monday", "tuesday", "wednesday"],
        "Load_Type": ["LIGHT_LOAD", "LIGHT_LOAD", None],
        "date": [
            "2020-01-01 10:00:00",
            "2020-01-04 12:00:00",
            "2020-01-05 08:30:00",
        ],
    }
    return pd.DataFrame(data)


def test_clean_data_types_and_missing_values(sample_raw_df, tmp_path):
    """All numeric/categorical/date fields are correctly typed and without NaNs."""
    pre = DataPreprocessor(raw_dir=tmp_path, processed_dir=tmp_path)

    df_clean = pre.clean_data(sample_raw_df)

    numeric_columns = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ]

    # Numeric columns: numeric dtype + no missing values
    for col in numeric_columns:
        assert np.issubdtype(
            df_clean[col].dtype, np.number
        ), f"{col} should be numeric"
        assert df_clean[col].isna().sum() == 0, f"{col} should not contain NaNs"

    # Date column: datetime dtype
    assert np.issubdtype(df_clean["date"].dtype, np.datetime64)

    # Categorical normalization: uppercase + no missing
    for col in ["WeekStatus", "Day_of_week", "Load_Type"]:
        assert df_clean[col].isna().sum() == 0
        assert (df_clean[col] == df_clean[col].str.upper()).all()

    # NSM should be int64 as enforced at the end of clean_data
    assert df_clean["NSM"].dtype == "int64"


def test_clean_data_respects_domain_rules(sample_raw_df, tmp_path):
    """
    Values that violate domain rules should be scaled / capped so that
    no column exceeds its configured max threshold.
    """
    pre = DataPreprocessor(raw_dir=tmp_path, processed_dir=tmp_path)
    df_clean = pre.clean_data(sample_raw_df)

    rules = {
        "Leading_Current_Power_Factor": 100,
        "Lagging_Current_Power_Factor": 100,
        "CO2(tCO2)": 0.02,
        "Lagging_Current_Reactive.Power_kVarh": 96.91,
        "Leading_Current_Reactive_Power_kVarh": 27.76,
        "NSM": 85500,
        "Usage_kWh": 157.18,
    }

    for col, max_val in rules.items():
        assert (df_clean[col] <= max_val).all(), (
            f"{col} exceeds expected max {max_val}"
        )


def test_run_creates_processed_file(tmp_path):
    """
    Integration-style test for run():
    - reads from raw_dir
    - cleans data
    - writes steel_energy_processed.csv to processed_dir
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    # Minimal realistic raw CSV
    df_raw = pd.DataFrame(
        {
            "Usage_kWh": [5.0, 15.0],
            "Lagging_Current_Reactive.Power_kVarh": [10.0, 20.0],
            "Leading_Current_Reactive_Power_kVarh": [1.0, 2.0],
            "CO2(tCO2)": [0.01, 0.03],
            "Lagging_Current_Power_Factor": [80.0, 90.0],
            "Leading_Current_Power_Factor": [85.0, 95.0],
            "NSM": [1000.0, 2000.0],
            "WeekStatus": ["WEEKDAY", "WEEKEND"],
            "Day_of_week": ["MONDAY", "SATURDAY"],
            "Load_Type": ["LIGHT_LOAD", "MEDIUM_LOAD"],
            "date": ["2020-01-01 10:00:00", "2020-01-02 11:00:00"],
        }
    )
    filename = "steel_energy_original.csv"
    df_raw.to_csv(raw_dir / filename, index=False)

    pre = DataPreprocessor(raw_dir=raw_dir, processed_dir=processed_dir)

    # Act
    df_processed = pre.run(filename=filename)

    # Assert
    output_path = processed_dir / "steel_energy_processed.csv"
    assert output_path.exists(), "Processed CSV was not created"

    df_from_disk = pd.read_csv(output_path)
    assert df_from_disk.shape == df_processed.shape
