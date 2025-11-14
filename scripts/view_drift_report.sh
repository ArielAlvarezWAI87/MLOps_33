#!/bin/bash
# View drift monitoring reports

REPORTS_DIR="data/monitoring/reports"

echo "=========================================="
echo "DRIFT MONITORING REPORTS"
echo "=========================================="
echo ""

if [ ! -d "$REPORTS_DIR" ]; then
    echo "❌ No reports found. Run drift monitoring first:"
    echo "   python src/monitoring/drift_simulator.py"
    echo "   python src/monitoring/run_drift_monitoring.py"
    exit 1
fi

# Show summary
if [ -f "$REPORTS_DIR/summary.json" ]; then
    echo "SUMMARY"
    echo "-------"
    python -c "
import json
with open('$REPORTS_DIR/summary.json') as f:
    data = json.load(f)
    summary = data['summary']
    baseline = data['baseline_metrics']

    print(f\"Total scenarios: {summary['total_scenarios']}\")
    print(f\"  🚨 ALERT: {summary['scenarios_with_alerts']}\")
    print(f\"  ⚠️  WARNING: {summary['scenarios_with_warnings']}\")
    print(f\"  ✓ OK: {summary['scenarios_ok']}\")
    print()
    print(f\"Baseline Performance:\")
    print(f\"  R² Score: {baseline['r2_score']:.4f}\")
    print(f\"  MAE: {baseline['mae']:.4f}\")
    print(f\"  RMSE: {baseline['rmse']:.4f}\")
    "
    echo ""
fi

# List individual reports
echo "INDIVIDUAL REPORTS"
echo "------------------"
ls -1 "$REPORTS_DIR"/report_*.json | while read report; do
    filename=$(basename "$report")
    scenario=$(echo "$filename" | sed 's/report_drift_//' | sed 's/.json//' | tr '_' ' ' | sed 's/\b\(.\)/\u\1/g')

    echo "• $scenario"
    python -c "
import json
with open('$report') as f:
    data = json.load(f)
    status = data['overall_status']
    perf = data['performance']
    drift = data['drift_detection']

    print(f\"  Status: {status}\")
    print(f\"  Performance: R²={perf['r2_score']:.4f}, MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}\")
    print(f\"  Drift: {drift['features_with_drift']}/{drift['total_features_checked']} features affected\")
    print()
    "
done

echo ""
echo "View detailed reports:"
echo "  cat $REPORTS_DIR/summary.json | python -m json.tool"
echo "  cat $REPORTS_DIR/report_drift_combined.json | python -m json.tool"
