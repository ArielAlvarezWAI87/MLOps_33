"""
drift_simulator.py
───────────────────────────────────────────────
Module: DriftSimulator
Purpose: Simulate different types of data drift for monitoring

Types of drift simulated:
1. Mean shift - Changes in feature means
2. Missing features - Simulate sensor failures
3. Seasonal changes - Simulate time-based patterns
4. Variance changes - Changes in feature variance
"""

import sys
from pathlib import Path

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DriftSimulator:
    """Simulate various types of data drift on datasets."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize DriftSimulator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def simulate_mean_shift(
        self,
        df: pd.DataFrame,
        features: List[str],
        shift_percentage: float = 20.0
    ) -> pd.DataFrame:
        """
        Simulate mean shift in specified features.

        Args:
            df: Original dataframe
            features: List of features to shift
            shift_percentage: Percentage to shift the mean (default 20%)

        Returns:
            DataFrame with shifted features
        """
        df_drift = df.copy()

        for feature in features:
            if feature in df_drift.columns:
                mean_val = df_drift[feature].mean()
                shift_amount = mean_val * (shift_percentage / 100.0)
                df_drift[feature] = df_drift[feature] + shift_amount
                logger.info(f"Mean shift applied to {feature}: +{shift_amount:.4f} ({shift_percentage}%)")

        return df_drift

    def simulate_missing_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        missing_percentage: float = 30.0
    ) -> pd.DataFrame:
        """
        Simulate missing values in features (sensor failures).

        Args:
            df: Original dataframe
            features: List of features to make missing
            missing_percentage: Percentage of values to make missing

        Returns:
            DataFrame with missing values
        """
        df_drift = df.copy()

        for feature in features:
            if feature in df_drift.columns:
                n_samples = len(df_drift)
                n_missing = int(n_samples * (missing_percentage / 100.0))
                missing_indices = np.random.choice(n_samples, n_missing, replace=False)
                df_drift.loc[missing_indices, feature] = np.nan
                logger.info(f"Missing values added to {feature}: {n_missing}/{n_samples} ({missing_percentage}%)")

        return df_drift

    def simulate_seasonal_change(
        self,
        df: pd.DataFrame,
        seasonal_multiplier: float = 1.3
    ) -> pd.DataFrame:
        """
        Simulate seasonal changes (e.g., higher energy consumption in winter).

        Args:
            df: Original dataframe
            seasonal_multiplier: Multiplier for energy-related features

        Returns:
            DataFrame with seasonal changes
        """
        df_drift = df.copy()

        energy_features = [
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'NSM'
        ]

        for feature in energy_features:
            if feature in df_drift.columns:
                df_drift[feature] = df_drift[feature] * seasonal_multiplier
                logger.info(f"Seasonal change applied to {feature}: ×{seasonal_multiplier}")

        return df_drift

    def simulate_variance_change(
        self,
        df: pd.DataFrame,
        features: List[str],
        variance_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Simulate change in variance (more volatile measurements).

        Args:
            df: Original dataframe
            features: List of features to modify
            variance_multiplier: Factor to multiply standard deviation

        Returns:
            DataFrame with changed variance
        """
        df_drift = df.copy()

        for feature in features:
            if feature in df_drift.columns:
                mean_val = df_drift[feature].mean()
                std_val = df_drift[feature].std()

                # Add noise proportional to new variance
                noise = np.random.normal(0, std_val * (variance_multiplier - 1), len(df_drift))
                df_drift[feature] = df_drift[feature] + noise

                logger.info(f"Variance change applied to {feature}: std ×{variance_multiplier}")

        return df_drift

    def simulate_combined_drift(
        self,
        df: pd.DataFrame,
        drift_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Simulate multiple types of drift combined.

        Args:
            df: Original dataframe
            drift_config: Configuration dict for drift types

        Returns:
            DataFrame with combined drift
        """
        if drift_config is None:
            # Default configuration
            drift_config = {
                'mean_shift': {
                    'features': ['Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor'],
                    'shift_percentage': 15.0
                },
                'missing_features': {
                    'features': ['NSM'],
                    'missing_percentage': 20.0
                },
                'seasonal_change': {
                    'seasonal_multiplier': 1.25
                }
            }

        df_drift = df.copy()

        if 'mean_shift' in drift_config:
            df_drift = self.simulate_mean_shift(
                df_drift,
                **drift_config['mean_shift']
            )

        if 'missing_features' in drift_config:
            df_drift = self.simulate_missing_features(
                df_drift,
                **drift_config['missing_features']
            )

        if 'seasonal_change' in drift_config:
            df_drift = self.simulate_seasonal_change(
                df_drift,
                **drift_config['seasonal_change']
            )

        if 'variance_change' in drift_config:
            df_drift = self.simulate_variance_change(
                df_drift,
                **drift_config['variance_change']
            )

        return df_drift


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MONITORING_DIR = ROOT_DIR / "data" / "monitoring"
    MONITORING_DIR.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("DATA DRIFT SIMULATION")
    print("="*80 + "\n")

    # Load original processed data
    print("Loading original data...")
    df_original = pd.read_csv(PROCESSED_DIR / "steel_energy_processed.csv", parse_dates=['date'])
    print(f"Original data shape: {df_original.shape}")

    # Initialize simulator
    simulator = DriftSimulator(random_seed=42)

    # Scenario 1: Mean shift (gradual drift)
    print("\n--- Scenario 1: Mean Shift (Gradual Drift) ---")
    df_mean_drift = simulator.simulate_mean_shift(
        df_original,
        features=['Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor'],
        shift_percentage=20.0
    )
    df_mean_drift.to_csv(MONITORING_DIR / "drift_mean_shift.csv", index=False)
    print(f"✓ Saved: {MONITORING_DIR / 'drift_mean_shift.csv'}")

    # Scenario 2: Missing features (sensor failure)
    print("\n--- Scenario 2: Missing Features (Sensor Failure) ---")
    df_missing_drift = simulator.simulate_missing_features(
        df_original,
        features=['NSM', 'CO2(tCO2)'],
        missing_percentage=30.0
    )
    df_missing_drift.to_csv(MONITORING_DIR / "drift_missing_features.csv", index=False)
    print(f"✓ Saved: {MONITORING_DIR / 'drift_missing_features.csv'}")

    # Scenario 3: Seasonal change
    print("\n--- Scenario 3: Seasonal Change (Winter Production) ---")
    df_seasonal_drift = simulator.simulate_seasonal_change(
        df_original,
        seasonal_multiplier=1.3
    )
    df_seasonal_drift.to_csv(MONITORING_DIR / "drift_seasonal.csv", index=False)
    print(f"✓ Saved: {MONITORING_DIR / 'drift_seasonal.csv'}")

    # Scenario 4: Combined drift (realistic scenario)
    print("\n--- Scenario 4: Combined Drift (Realistic) ---")
    df_combined_drift = simulator.simulate_combined_drift(df_original)
    df_combined_drift.to_csv(MONITORING_DIR / "drift_combined.csv", index=False)
    print(f"✓ Saved: {MONITORING_DIR / 'drift_combined.csv'}")

    print("\n" + "="*80)
    print("✅ Data drift simulation completed!")
    print(f"Drift datasets saved to: {MONITORING_DIR}")
    print("="*80 + "\n")
