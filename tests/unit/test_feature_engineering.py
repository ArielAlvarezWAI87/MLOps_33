"""
test_feature_engineering.py
────────────────────────────────────────
Unit / integration tests for src.features.feature_engineering.FeatureEngineer

"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_clean_df():
    """
    Minimal cleaned dataset, consistent with DataPreprocessor.clean_data()
    output — used as input for FeatureEngineer.
    """
    dates = pd.to_datetime(
        [
            "2020-01-01 10:00:00",
            "2020-01-02 12:00:00",
            "2020-01-05 08:30:00",
        ]
    )
    data = {
        "date": dates,
        "Usage_kWh": [10.0, 20.0, 30.0],
        "Lagging_Current_Reactive.Power_kVarh": [5.0, 10.0, 15.0],
        "Leading_Current_Reactive_Power_kVarh": [1.0, 2.0, 3.0],
        "CO2(tCO2)": [0.01, 0.02, 0.015],
        "Lagging_Current_Power_Factor": [0.8, 0.9, 0.85],
        "Leading_Current_Power_Factor": [0.75, 0.88, 0.8],
        "NSM": [1000, 2000, 3000],
        "WeekStatus": ["WEEKDAY", "WEEKDAY", "WEEKEND"],
        "Day_of_week": ["MONDAY", "TUESDAY", "SUNDAY"],
        "Load_Type": ["LIGHT_LOAD", "MEDIUM_LOAD", "MAXIMUM_LOAD"],
    }
    return pd.DataFrame(data)


def test_engineer_features_creates_expected_columns(sample_clean_df, tmp_path):
    """Check that all engineered columns are created and have consistent logic."""
    fe = FeatureEngineer(raw_processed_dir=tmp_path, save_dir=tmp_path)

    df_eng = fe.engineer_features(sample_clean_df)

    expected_new_cols = {
        "year",
        "month",
        "day",
        "hour",
        "day_of_week_num",
        "quarter",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "power_factor_ratio",
        "power_factor_diff",
        "avg_power_factor",
        "reactive_power_total",
        "reactive_power_diff",
        "reactive_power_ratio",
        "co2_per_kwh",
        "is_high_consumption",
        "nsm_per_kwh",
    }

    missing = expected_new_cols.difference(df_eng.columns)
    assert not missing, f"Missing engineered columns: {missing}"

    # is_weekend should be 1 only when day_of_week_num >= 5
    computed_weekend = (df_eng["day_of_week_num"] >= 5).astype(int)
    assert (df_eng["is_weekend"] == computed_weekend).all()

    # Cyclical features should be in [-1, 1]
    cyc_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
    for col in cyc_cols:
        assert df_eng[col].between(-1.000001, 1.000001).all(), (
            f"{col} out of [-1, 1] range"
        )

    # Example check for co2_per_kwh ratio (row 0)
    expected_ratio = sample_clean_df.loc[0, "CO2(tCO2)"] / (
        sample_clean_df.loc[0, "Usage_kWh"] + 1e-6
    )
    assert np.isclose(df_eng.loc[0, "co2_per_kwh"], expected_ratio)


def test_build_preprocessing_pipeline_fit_transform(sample_clean_df, tmp_path):
    """
    The preprocessing pipeline should:
    - fit successfully on engineered features
    - impute missing values
    - return a finite numeric matrix
    """
    fe = FeatureEngineer(raw_processed_dir=tmp_path, save_dir=tmp_path)
    df_eng = fe.engineer_features(sample_clean_df)

    # Introduce some missing values to verify imputers work
    df_eng.loc[0, "Lagging_Current_Reactive.Power_kVarh"] = np.nan
    df_eng.loc[1, "WeekStatus"] = np.nan
    df_eng.loc[2, "Load_Type"] = np.nan

    preprocessor = fe.build_preprocessing_pipeline(df_eng)

    X_processed = preprocessor.fit_transform(df_eng)

    assert X_processed.shape[0] == df_eng.shape[0]
    assert np.isfinite(X_processed).all(), (
        "Preprocessed matrix contains NaN or infinite values"
    )


def test_run_end_to_end_creates_artifacts(tmp_path, sample_clean_df):
    """
    Integration-style test for FeatureEngineer.run():
    - reads steel_energy_processed.csv
    - engineers features + builds preprocessor
    - writes X_features.csv, y_target.csv, preprocessor.pkl, feature_info.pkl
    """
    filename = "steel_energy_processed.csv"
    sample_clean_df.to_csv(tmp_path / filename, index=False)

    fe = FeatureEngineer(raw_processed_dir=tmp_path, save_dir=tmp_path)

    X_processed, y, preprocessor = fe.run(filename=filename)

    # Check that outputs were created
    X_path = tmp_path / "X_features.csv"
    y_path = tmp_path / "y_target.csv"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    feature_info_path = tmp_path / "feature_info.pkl"

    assert X_path.exists()
    assert y_path.exists()
    assert preprocessor_path.exists()
    assert feature_info_path.exists()

    # Basic shape sanity checks
    assert len(y) == sample_clean_df.shape[0]
    assert X_processed.shape[0] == sample_clean_df.shape[0]
    assert preprocessor is not None
