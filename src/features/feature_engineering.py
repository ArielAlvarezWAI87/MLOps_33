"""
feature_engineering.py
───────────────────────────────────────────────
Module: FeatureEngineer
Author: Bruno Sánchez García
Purpose:
    Create engineered features and build preprocessing
    pipelines for model training.

Outputs:
    ../data/processed/X_features.csv
    ../data/processed/y_target.csv
    ../data/processed/preprocessor.pkl
    ../data/processed/feature_info.pkl
"""

import sys
from pathlib import Path

# Add project root to path (allows running script directly)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings("ignore")

# Import reproducibility utilities
from src.utils.reproducibility import set_seeds, RANDOM_SEED


class FeatureEngineer:
    """Adds engineered features and constructs preprocessing pipelines."""

    def __init__(self,
                 raw_processed_dir: str = "../../data/processed",
                 save_dir: str = "../../data/processed"):
        """
        Initialize directories for reading processed data and saving outputs.

        Args:
            raw_processed_dir (str): Path where the cleaned CSV is stored.
            save_dir (str): Path to save the engineered and transformed data.
        """
        self.raw_processed_dir = Path(raw_processed_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def load_clean_data(self, filename: str = "steel_energy_processed.csv") -> pd.DataFrame:
        """Load cleaned dataset."""
        path = self.raw_processed_dir / filename
        print(f"\n📂 Loading cleaned dataset from: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        print(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    # ----------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal, cyclical, and energy efficiency features."""
        print("\n" + "=" * 80)
        print("STEP 1: FEATURE ENGINEERING")
        print("=" * 80)

        df_eng = df.copy()

        # Temporal features
        df_eng["year"] = df_eng["date"].dt.year
        df_eng["month"] = df_eng["date"].dt.month
        df_eng["day"] = df_eng["date"].dt.day
        df_eng["hour"] = df_eng["date"].dt.hour
        df_eng["day_of_week_num"] = df_eng["date"].dt.dayofweek
        df_eng["quarter"] = df_eng["date"].dt.quarter
        df_eng["is_weekend"] = (df_eng["day_of_week_num"] >= 5).astype(int)

        # Cyclical features
        df_eng["hour_sin"] = np.sin(2 * np.pi * df_eng["hour"] / 24)
        df_eng["hour_cos"] = np.cos(2 * np.pi * df_eng["hour"] / 24)
        df_eng["month_sin"] = np.sin(2 * np.pi * df_eng["month"] / 12)
        df_eng["month_cos"] = np.cos(2 * np.pi * df_eng["month"] / 12)
        df_eng["dow_sin"] = np.sin(2 * np.pi * df_eng["day_of_week_num"] / 7)
        df_eng["dow_cos"] = np.cos(2 * np.pi * df_eng["day_of_week_num"] / 7)

        # Power factor features
        df_eng["power_factor_ratio"] = df_eng["Lagging_Current_Power_Factor"] / (
            df_eng["Leading_Current_Power_Factor"] + 1e-6
        )
        df_eng["power_factor_diff"] = (
            df_eng["Lagging_Current_Power_Factor"] - df_eng["Leading_Current_Power_Factor"]
        )
        df_eng["avg_power_factor"] = (
            df_eng["Lagging_Current_Power_Factor"] + df_eng["Leading_Current_Power_Factor"]
        ) / 2

        # Reactive power features
        df_eng["reactive_power_total"] = (
            df_eng["Lagging_Current_Reactive.Power_kVarh"] +
            df_eng["Leading_Current_Reactive_Power_kVarh"]
        )
        df_eng["reactive_power_diff"] = (
            df_eng["Lagging_Current_Reactive.Power_kVarh"] -
            df_eng["Leading_Current_Reactive_Power_kVarh"]
        )
        df_eng["reactive_power_ratio"] = (
            df_eng["Lagging_Current_Reactive.Power_kVarh"] /
            (df_eng["Leading_Current_Reactive_Power_kVarh"] + 1e-6)
        )

        # Energy efficiency indicators
        df_eng["co2_per_kwh"] = df_eng["CO2(tCO2)"] / (df_eng["Usage_kWh"] + 1e-6)
        df_eng["is_high_consumption"] = (
            df_eng["Usage_kWh"] > df_eng["Usage_kWh"].median()
        ).astype(int)
        df_eng["nsm_per_kwh"] = df_eng["NSM"] / (df_eng["Usage_kWh"] + 1e-6)

        print("✓ Engineered features added successfully")
        print(f"New columns added: {df_eng.shape[1] - df.shape[1]}")
        return df_eng

    # ----------------------------------------------------------------------
    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline for numeric and categorical features."""
        print("\n" + "=" * 80)
        print("STEP 2: BUILD PREPROCESSING PIPELINE")
        print("=" * 80)

        # Define column groups
        num_skew = [
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "reactive_power_total",
            "NSM",
            "co2_per_kwh",
            "nsm_per_kwh",
        ]
        num_lin = [
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "hour", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "power_factor_ratio", "power_factor_diff", "avg_power_factor",
            "reactive_power_diff", "reactive_power_ratio",
            "year", "month", "day", "quarter",
            "day_of_week_num", "is_weekend", "is_high_consumption",
        ]
        cat_nom = ["WeekStatus"]
        cat_ord = ["Load_Type"]

        # Filter only existing columns
        num_skew = [c for c in num_skew if c in X.columns]
        num_lin = [c for c in num_lin if c in X.columns]
        cat_nom = [c for c in cat_nom if c in X.columns]
        cat_ord = [c for c in cat_ord if c in X.columns]

        print(f"Skewed numeric: {num_skew}")
        print(f"Linear numeric: {num_lin}")
        print(f"Nominal categorical: {cat_nom}")
        print(f"Ordinal categorical: {cat_ord}")

        # Define sub-pipelines
        num_skew_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", MinMaxScaler())
        ])

        num_lin_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])

        nom_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False,
                                     handle_unknown="ignore",
                                     drop="first"))
        ])

        ord_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                categories=[["LIGHT_LOAD", "MEDIUM_LOAD", "MAXIMUM_LOAD"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ("num_skew", num_skew_pipe, num_skew),
            ("num_lin", num_lin_pipe, num_lin),
            ("cat_nom", nom_pipe, cat_nom),
            ("cat_ord", ord_pipe, cat_ord)
        ], remainder="drop")

        print("✓ Preprocessing pipeline built successfully")
        return preprocessor

    # ----------------------------------------------------------------------
    def run(self, filename: str = "steel_energy_processed.csv"):
        """Full feature engineering pipeline: load, transform, and save."""
        # Set random seeds for reproducibility
        set_seeds(RANDOM_SEED)

        df = self.load_clean_data(filename)
        df_eng = self.engineer_features(df)

        # Separate features and target
        exclude_cols = ["date", "Usage_kWh", "Day_of_week"]
        feature_cols = [c for c in df_eng.columns if c not in exclude_cols]
        X = df_eng[feature_cols].copy()
        y = df_eng["Usage_kWh"].copy()

        # Build and fit preprocessor
        preprocessor = self.build_preprocessing_pipeline(X)
        X_processed = preprocessor.fit_transform(X)

        # Convert to DataFrame for better interpretability
        X_processed = pd.DataFrame(
            X_processed,
            columns=preprocessor.get_feature_names_out(),
            index=X.index
        )

        print(f"\n✓ Transformation complete: {X.shape} → {X_processed.shape}")

        # Save outputs
        X_processed.to_csv(self.save_dir / "X_features.csv", index=False)
        y.to_csv(self.save_dir / "y_target.csv", index=False)
        joblib.dump(preprocessor, self.save_dir / "preprocessor.pkl")
        joblib.dump({
            "num_skew": preprocessor.transformers_[0][2],
            "num_lin": preprocessor.transformers_[1][2],
            "cat_nom": preprocessor.transformers_[2][2],
            "cat_ord": preprocessor.transformers_[3][2]
        }, self.save_dir / "feature_info.pkl")

        print("\n💾 Saved:")
        print(f"  • X_features.csv ({X_processed.shape[0]} rows × {X_processed.shape[1]} cols)")
        print(f"  • y_target.csv ({y.shape[0]} rows)")
        print("  • preprocessor.pkl")
        print("  • feature_info.pkl")
        print("\n✅ Feature engineering pipeline completed successfully!")

        return X_processed, y, preprocessor



# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    #root="C:/Users/AABDC5/repo/PythonML/dev/mlops-homework/"
    ROOT_DIR = Path(__file__).resolve().parents[2] 
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"

    fe = FeatureEngineer(
        raw_processed_dir = PROCESSED_DIR,
        save_dir = PROCESSED_DIR
    )
    fe.run("steel_energy_processed.csv")
