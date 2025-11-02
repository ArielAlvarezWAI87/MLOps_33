"""
preprocessing.py
───────────────────────────────────────────────
Module: DataPreprocessor
Author: Bruno Sánchez García
Purpose:
    Clean and prepare raw steel energy data for downstream
    feature engineering and modeling.

Output:
    ../data/processed/steel_energy_processed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from pathlib import Path


class DataPreprocessor:
    """Handles loading, cleaning, and preparation of raw energy data."""

    def __init__(self, raw_dir: str = "../../data/raw",
                 processed_dir: str = "../../data/processed"):
        """
        Initialize paths for input/output directories.

        Args:
            raw_dir (str): Directory containing raw CSV files.
            processed_dir (str): Directory where processed file will be saved.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def load_data(self, filename: str = "steel_energy_original.csv") -> pd.DataFrame:
        """Load raw dataset and return a DataFrame."""
        path = self.raw_dir / filename
        print(f"\n📂 Loading data from: {path}")
        df = pd.read_csv(path)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    # ----------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform all cleaning steps and return cleaned DataFrame."""

        print("\n" + "=" * 80)
        print("STEP 1: TYPE CONVERSIONS")
        print("=" * 80)

        numeric_columns = [
            "Usage_kWh",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "NSM",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        print("✓ Numeric and datetime conversion completed.")

        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 2: HANDLE CATEGORICAL VARIABLES")
        print("=" * 80)

        categorical_cols = ["WeekStatus", "Day_of_week", "Load_Type"]

        for col in categorical_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .replace(["NAN", "NA", "NONE", "NULL", ""], np.nan)
                )
                mode_value = df[col].mode().iloc[0] if df[col].mode().size > 0 else "UNKNOWN"
                df[col] = df[col].fillna(mode_value)
                print(f"✓ {col}: filled missing with mode '{mode_value}'")

        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 3: HANDLE MISSING NUMERIC VALUES")
        print("=" * 80)

        # Replace NaN in Usage_kWh with median per Load_Type
        if "Usage_kWh" in df.columns and "Load_Type" in df.columns:
            medians = df.groupby("Load_Type")["Usage_kWh"].median()
            df["Usage_kWh"] = df.apply(
                lambda r: medians.get(r["Load_Type"], df["Usage_kWh"].median())
                if pd.isna(r["Usage_kWh"])
                else r["Usage_kWh"],
                axis=1,
            )

        # Fill remaining numeric NaNs with 0
        for col in numeric_columns:
            if col != "Usage_kWh" and col in df.columns:
                df[col] = df[col].fillna(0)

        print("✓ Numeric NaNs handled (median for Usage_kWh, 0 for others)")

        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 4: APPLY DATA QUALITY RULES")
        print("=" * 80)

        # Rule set based on domain constraints
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
            if col not in df.columns:
                continue
            initial_violations = (df[col] > max_val).sum()

            # Apply scaling rules
            if col in ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]:
                df.loc[df[col] > max_val, col] /= 100
            elif col == "CO2(tCO2)":
                df.loc[df[col] > 0.2, col] /= 100000
            elif col == "Lagging_Current_Reactive.Power_kVarh":
                df.loc[df[col] > 100, col] /= 100
            elif col == "Leading_Current_Reactive_Power_kVarh":
                df.loc[df[col] > max_val, col] /= 1000
            elif col == "NSM":
                df.loc[df[col] > max_val, col] /= 100
            elif col == "Usage_kWh":
                df.loc[df[col] > max_val, col] /= 1000

            # Cap any remaining excessive values
            df.loc[df[col] > max_val, col] = max_val

            final_violations = (df[col] > max_val).sum()
            print(
                f"✓ {col}: {initial_violations} values above {max_val}, "
                f"after correction {final_violations}"
            )

        df["NSM"] = df["NSM"].astype("int64")

        print("✓ All data quality rules applied successfully.")

        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 5: FINAL CHECKS AND SUMMARY")
        print("=" * 80)

        print(f"Final shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print("Sample rows:\n", df.head(3))

        return df

    # ----------------------------------------------------------------------
    def save_processed(self, df: pd.DataFrame,
                       filename: str = "steel_energy_processed.csv") -> None:
        """Save the cleaned DataFrame to the processed directory."""
        path = self.processed_dir / filename
        df.to_csv(path, index=False)
        print(f"\n💾 Cleaned dataset saved to: {path}")

    # ----------------------------------------------------------------------
    def run(self, filename: str = "steel_energy_original.csv") -> pd.DataFrame:
        """Full pipeline: load, clean, and save data."""
        df = self.load_data(filename)
        df_clean = self.clean_data(df)
        self.save_processed(df_clean)
        print("\n✅ Data preprocessing completed successfully!")
        return df_clean


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    #root= "C:/Users/AABDC5/repo/PythonML/dev/mlops-homework/"
    #root = sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    ROOT_DIR = Path(__file__).resolve().parents[2] 
    RAW_DIR = ROOT_DIR / "data" / "raw"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"

    preprocessor = DataPreprocessor(
        raw_dir = RAW_DIR,
        processed_dir = PROCESSED_DIR
    )

    df_clean = preprocessor.run("steel_energy_original.csv")
    #print(ROOT_DIR,RAW_DIR,PROCESSED_DIR)
    #print(sys.path)




