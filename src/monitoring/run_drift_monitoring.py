"""
run_drift_monitoring.py
───────────────────────────────────────────────
Script: Run drift monitoring and performance evaluation

Evaluates model performance on drifted datasets and generates reports.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from src.monitoring.drift_detector import DriftDetector
from src.utils.reproducibility import set_seeds, RANDOM_SEED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Set seeds for reproducibility
    set_seeds(RANDOM_SEED)

    # Paths
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MONITORING_DIR = ROOT_DIR / "data" / "monitoring"
    REPORTS_DIR = MONITORING_DIR / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = ROOT_DIR / "models" / "rulefit.pkl"
    PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.pkl"

    print("\n" + "="*80)
    print("DRIFT MONITORING AND PERFORMANCE EVALUATION")
    print("="*80 + "\n")

    # Load baseline data
    print("Step 1: Loading baseline data...")
    df_baseline = pd.read_csv(
        PROCESSED_DIR / "steel_energy_processed.csv",
        parse_dates=['date']
    )
    print(f"✓ Baseline data loaded: {df_baseline.shape}")

    # Calculate baseline metrics
    print("\nStep 2: Calculating baseline metrics...")
    detector = DriftDetector()

    baseline_metrics = detector.evaluate_model_performance(
        MODEL_PATH,
        PREPROCESSOR_PATH,
        df_baseline
    )

    print("✓ Baseline metrics:")
    print(f"  R² Score: {baseline_metrics['r2_score']:.4f}")
    print(f"  MAE: {baseline_metrics['mae']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")

    # Set baseline for comparisons
    detector.baseline_metrics = baseline_metrics

    # Drift scenarios to evaluate
    drift_scenarios = [
        ('drift_mean_shift.csv', 'Mean Shift (Gradual Drift)'),
        ('drift_missing_features.csv', 'Missing Features (Sensor Failure)'),
        ('drift_seasonal.csv', 'Seasonal Change (Winter Production)'),
        ('drift_combined.csv', 'Combined Drift (Realistic Scenario)')
    ]

    all_reports = []

    # Evaluate each drift scenario
    print("\nStep 3: Evaluating drift scenarios...\n")

    for filename, scenario_name in drift_scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*80}")

        drift_file = MONITORING_DIR / filename

        if not drift_file.exists():
            logger.warning(f"File not found: {drift_file}")
            continue

        # Load drifted data
        df_drift = pd.read_csv(drift_file, parse_dates=['date'])
        print(f"✓ Loaded drifted data: {df_drift.shape}")

        # Detect feature drift
        print("\n  Feature Drift Detection:")
        drift_results = detector.detect_feature_drift(
            df_baseline,
            df_drift
        )

        drifted_features = [f for f, r in drift_results.items() if r['has_drift']]
        if drifted_features:
            print(f"  ⚠️  Drift detected in {len(drifted_features)} features:")
            for feature in drifted_features[:5]:  # Show first 5
                result = drift_results[feature]
                print(f"    - {feature}: PSI={result['psi']:.3f}, " +
                      f"KS={result['ks_statistic']:.3f}, " +
                      f"Severity={result['drift_severity']}")
        else:
            print("  ✓ No significant drift detected")

        # Evaluate model performance
        print("\n  Model Performance:")
        current_metrics = detector.evaluate_model_performance(
            MODEL_PATH,
            PREPROCESSOR_PATH,
            df_drift
        )

        print(f"    R² Score: {current_metrics['r2_score']:.4f} " +
              f"(baseline: {baseline_metrics['r2_score']:.4f})")
        print(f"    MAE: {current_metrics['mae']:.4f} " +
              f"(baseline: {baseline_metrics['mae']:.4f})")
        print(f"    RMSE: {current_metrics['rmse']:.4f} " +
              f"(baseline: {baseline_metrics['rmse']:.4f})")

        # Compare with baseline
        print("\n  Baseline Comparison:")
        comparison = detector.compare_with_baseline(current_metrics)

        if comparison['has_degradation']:
            print("  🚨 PERFORMANCE DEGRADATION DETECTED!")
            for alert in comparison['alerts']:
                print(f"    [{alert['severity']}] {alert['metric']}: " +
                      f"{alert.get('change_pct', alert.get('change', 0)):.2f}% change")
                print(f"      → Action: {alert['action']}")
        else:
            print("  ✓ Performance within acceptable range")

        # Generate report
        report = detector.generate_report(
            drift_results,
            current_metrics,
            comparison,
            scenario_name
        )

        # Save report
        report_file = REPORTS_DIR / f"report_{filename.replace('.csv', '.json')}"
        detector.save_report(report, report_file)
        print(f"\n  ✓ Report saved: {report_file.name}")

        # Show recommended actions
        print("\n  Recommended Actions:")
        for i, action in enumerate(report['recommended_actions'], 1):
            print(f"    {i}. {action}")

        all_reports.append(report)

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    summary = {
        'total_scenarios': len(all_reports),
        'scenarios_with_alerts': sum(1 for r in all_reports if r['overall_status'] == 'ALERT'),
        'scenarios_with_warnings': sum(1 for r in all_reports if r['overall_status'] == 'WARNING'),
        'scenarios_ok': sum(1 for r in all_reports if r['overall_status'] == 'OK'),
    }

    print(f"Total scenarios evaluated: {summary['total_scenarios']}")
    print(f"  🚨 ALERT: {summary['scenarios_with_alerts']}")
    print(f"  ⚠️  WARNING: {summary['scenarios_with_warnings']}")
    print(f"  ✓ OK: {summary['scenarios_ok']}")

    # Save summary
    import json

    # Use the detector's save method to handle type conversion
    summary_data = {
        'summary': summary,
        'baseline_metrics': baseline_metrics,
        'reports': all_reports
    }
    detector.save_report(summary_data, REPORTS_DIR / "summary.json")

    print(f"\n✓ Summary saved: {REPORTS_DIR / 'summary.json'}")
    print("\n" + "="*80)
    print("✅ Drift monitoring completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
