"""
drift_detector.py
───────────────────────────────────────────────
Module: DriftDetector
Purpose: Detect data drift and performance degradation

Implements:
- Statistical drift detection (KS test, PSI)
- Performance monitoring
- Alert thresholds
- Recommended actions
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
from typing import Dict, List, Tuple, Optional
from scipy import stats
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift and monitor model performance."""

    # Alert thresholds
    THRESHOLDS = {
        'ks_statistic': 0.1,      # Kolmogorov-Smirnov test threshold
        'psi': 0.2,                # Population Stability Index threshold
        'r2_drop': 0.1,            # R² drop threshold (10% degradation)
        'mae_increase': 0.15,      # MAE increase threshold (15% increase)
        'rmse_increase': 0.15,     # RMSE increase threshold (15% increase)
        'missing_data': 0.05       # Missing data threshold (5%)
    }

    def __init__(self, baseline_metrics: Optional[Dict] = None):
        """
        Initialize DriftDetector.

        Args:
            baseline_metrics: Dict with baseline performance metrics
        """
        self.baseline_metrics = baseline_metrics or {}
        self.drift_reports = []

    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much a distribution has shifted.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        breakpoints = np.linspace(
            reference.min(),
            reference.max(),
            bins + 1
        )

        # Bin both distributions
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        curr_counts = np.histogram(current, bins=breakpoints)[0]

        # Avoid division by zero
        ref_percents = (ref_counts + 1) / (len(reference) + bins)
        curr_percents = (curr_counts + 1) / (len(current) + bins)

        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))

        return psi

    def detect_feature_drift(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect drift in features using statistical tests.

        Args:
            df_reference: Reference (baseline) dataset
            df_current: Current (monitoring) dataset
            features: List of features to check (None = all numeric)

        Returns:
            Dict with drift detection results
        """
        if features is None:
            features = df_reference.select_dtypes(include=[np.number]).columns.tolist()

        drift_results = {}

        for feature in features:
            if feature not in df_current.columns:
                logger.warning(f"Feature {feature} not in current dataset")
                continue

            # Get non-null values
            ref_values = df_reference[feature].dropna()
            curr_values = df_current[feature].dropna()

            # KS test (Kolmogorov-Smirnov)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)

            # PSI (Population Stability Index)
            psi = self.calculate_psi(ref_values, curr_values)

            # Missing data percentage
            missing_pct = df_current[feature].isna().sum() / len(df_current)

            # Determine drift status
            has_drift = (
                ks_stat > self.THRESHOLDS['ks_statistic'] or
                psi > self.THRESHOLDS['psi'] or
                missing_pct > self.THRESHOLDS['missing_data']
            )

            drift_results[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'psi': float(psi),
                'missing_pct': float(missing_pct),
                'has_drift': has_drift,
                'drift_severity': self._classify_drift_severity(ks_stat, psi, missing_pct)
            }

        return drift_results

    def _classify_drift_severity(
        self,
        ks_stat: float,
        psi: float,
        missing_pct: float
    ) -> str:
        """Classify drift severity level."""
        if missing_pct > 0.2:
            return "CRITICAL"
        elif ks_stat > 0.3 or psi > 0.4:
            return "SEVERE"
        elif ks_stat > self.THRESHOLDS['ks_statistic'] or psi > self.THRESHOLDS['psi']:
            return "MODERATE"
        else:
            return "NONE"

    def evaluate_model_performance(
        self,
        model_path: Path,
        preprocessor_path: Path,
        df: pd.DataFrame,
        target_col: str = 'Usage_kWh'
    ) -> Dict:
        """
        Evaluate model performance on dataset.

        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor
            df: Dataset to evaluate on
            target_col: Target column name

        Returns:
            Dict with performance metrics
        """
        from src.features.feature_engineering import FeatureEngineer

        # Load model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        # Engineer features
        engineer = FeatureEngineer()
        df_eng = engineer.engineer_features(df)

        # Prepare features
        exclude_cols = ["date", target_col, "Day_of_week"]
        feature_cols = [c for c in df_eng.columns if c not in exclude_cols]
        X = df_eng[feature_cols]
        y_true = df_eng[target_col]

        # Transform and predict
        X_processed = preprocessor.transform(X)
        y_pred = model.predict(X_processed)

        # Calculate metrics
        metrics = {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'n_samples': len(y_true),
            'n_missing': int(df.isnull().sum().sum())
        }

        return metrics

    def compare_with_baseline(
        self,
        current_metrics: Dict
    ) -> Dict:
        """
        Compare current metrics with baseline.

        Args:
            current_metrics: Current performance metrics

        Returns:
            Dict with comparison results and alerts
        """
        if not self.baseline_metrics:
            logger.warning("No baseline metrics set")
            return {'alerts': [], 'degradation': {}}

        alerts = []
        degradation = {}

        # R² degradation
        if 'r2_score' in self.baseline_metrics and 'r2_score' in current_metrics:
            r2_drop = self.baseline_metrics['r2_score'] - current_metrics['r2_score']
            degradation['r2_drop'] = float(r2_drop)

            if r2_drop > self.THRESHOLDS['r2_drop']:
                alerts.append({
                    'metric': 'R² Score',
                    'severity': 'HIGH' if r2_drop > 0.2 else 'MEDIUM',
                    'baseline': self.baseline_metrics['r2_score'],
                    'current': current_metrics['r2_score'],
                    'change': -r2_drop,
                    'threshold': self.THRESHOLDS['r2_drop'],
                    'action': 'RETRAIN_MODEL'
                })

        # MAE increase
        if 'mae' in self.baseline_metrics and 'mae' in current_metrics:
            mae_change_pct = (
                (current_metrics['mae'] - self.baseline_metrics['mae']) /
                self.baseline_metrics['mae']
            )
            degradation['mae_change_pct'] = float(mae_change_pct)

            if mae_change_pct > self.THRESHOLDS['mae_increase']:
                alerts.append({
                    'metric': 'MAE',
                    'severity': 'HIGH' if mae_change_pct > 0.3 else 'MEDIUM',
                    'baseline': self.baseline_metrics['mae'],
                    'current': current_metrics['mae'],
                    'change_pct': mae_change_pct * 100,
                    'threshold': self.THRESHOLDS['mae_increase'] * 100,
                    'action': 'REVIEW_FEATURE_PIPELINE'
                })

        # RMSE increase
        if 'rmse' in self.baseline_metrics and 'rmse' in current_metrics:
            rmse_change_pct = (
                (current_metrics['rmse'] - self.baseline_metrics['rmse']) /
                self.baseline_metrics['rmse']
            )
            degradation['rmse_change_pct'] = float(rmse_change_pct)

            if rmse_change_pct > self.THRESHOLDS['rmse_increase']:
                alerts.append({
                    'metric': 'RMSE',
                    'severity': 'HIGH' if rmse_change_pct > 0.3 else 'MEDIUM',
                    'baseline': self.baseline_metrics['rmse'],
                    'current': current_metrics['rmse'],
                    'change_pct': rmse_change_pct * 100,
                    'threshold': self.THRESHOLDS['rmse_increase'] * 100,
                    'action': 'INVESTIGATE_DATA_QUALITY'
                })

        return {
            'alerts': alerts,
            'degradation': degradation,
            'has_degradation': len(alerts) > 0
        }

    def generate_report(
        self,
        drift_results: Dict,
        performance_metrics: Dict,
        comparison: Dict,
        scenario_name: str
    ) -> Dict:
        """
        Generate comprehensive drift detection report.

        Args:
            drift_results: Feature drift detection results
            performance_metrics: Performance metrics
            comparison: Baseline comparison
            scenario_name: Name of the drift scenario

        Returns:
            Complete drift report
        """
        # Count features with drift
        drifted_features = [
            f for f, r in drift_results.items()
            if r['has_drift']
        ]

        # Determine overall status
        if comparison['has_degradation']:
            overall_status = "ALERT"
        elif len(drifted_features) > 0:
            overall_status = "WARNING"
        else:
            overall_status = "OK"

        report = {
            'scenario': scenario_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_status': overall_status,
            'drift_detection': {
                'total_features_checked': len(drift_results),
                'features_with_drift': len(drifted_features),
                'drifted_features': drifted_features,
                'feature_details': drift_results
            },
            'performance': performance_metrics,
            'baseline_comparison': comparison,
            'recommended_actions': self._get_recommended_actions(
                drift_results,
                comparison
            )
        }

        return report

    def _get_recommended_actions(
        self,
        drift_results: Dict,
        comparison: Dict
    ) -> List[str]:
        """Determine recommended actions based on drift and performance."""
        actions = []

        # Check for critical drift
        critical_features = [
            f for f, r in drift_results.items()
            if r.get('drift_severity') == 'CRITICAL'
        ]
        if critical_features:
            actions.append(f"URGENT: Investigate data quality issues in {', '.join(critical_features)}")

        # Check for performance degradation
        if comparison.get('has_degradation'):
            for alert in comparison.get('alerts', []):
                if alert['severity'] == 'HIGH':
                    actions.append(f"HIGH PRIORITY: {alert['action']} - {alert['metric']} degraded significantly")
                else:
                    actions.append(f"MEDIUM PRIORITY: {alert['action']} - {alert['metric']} showing degradation")

        # Check for moderate drift
        moderate_features = [
            f for f, r in drift_results.items()
            if r.get('drift_severity') in ['MODERATE', 'SEVERE']
        ]
        if moderate_features and not actions:
            actions.append(f"Monitor features: {', '.join(moderate_features)} - showing drift patterns")

        if not actions:
            actions.append("No action required - system operating normally")

        return actions

    def save_report(self, report: Dict, output_path: Path):
        """Save drift report to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj

        report_clean = convert_types(report)

        with open(output_path, 'w') as f:
            json.dump(report_clean, f, indent=2)
        logger.info(f"Report saved to: {output_path}")
