#!/bin/bash
# Verify reproducibility by running pipelines twice and comparing outputs

set -e

echo "=========================================="
echo "REPRODUCIBILITY VERIFICATION TEST"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create temporary directories for two runs
RUN1_DIR="temp_run1"
RUN2_DIR="temp_run2"

echo "Step 1: Clean up any previous test runs..."
rm -rf $RUN1_DIR $RUN2_DIR
mkdir -p $RUN1_DIR $RUN2_DIR

echo ""
echo "Step 2: Running pipeline - First execution..."
python -c "
from pathlib import Path
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.rulefit_trainer import RuleFitTrainer
import pandas as pd
import numpy as np

ROOT = Path.cwd()
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / '$RUN1_DIR'

# Preprocessing
preprocessor = DataPreprocessor(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
df_clean = preprocessor.run('steel_energy_original.csv')

# Feature Engineering
fe = FeatureEngineer(raw_processed_dir=PROCESSED_DIR, save_dir=PROCESSED_DIR)
X, y, preprocessor = fe.run('steel_energy_processed.csv')

print(f'\nRun 1 completed:')
print(f'  X_features shape: {X.shape}')
print(f'  First 5 values of first feature: {X.iloc[:5, 0].values}')
"

echo ""
echo "Step 3: Running pipeline - Second execution..."
python -c "
from pathlib import Path
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.rulefit_trainer import RuleFitTrainer
import pandas as pd
import numpy as np

ROOT = Path.cwd()
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / '$RUN2_DIR'

# Preprocessing
preprocessor = DataPreprocessor(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
df_clean = preprocessor.run('steel_energy_original.csv')

# Feature Engineering
fe = FeatureEngineer(raw_processed_dir=PROCESSED_DIR, save_dir=PROCESSED_DIR)
X, y, preprocessor = fe.run('steel_energy_processed.csv')

print(f'\nRun 2 completed:')
print(f'  X_features shape: {X.shape}')
print(f'  First 5 values of first feature: {X.iloc[:5, 0].values}')
"

echo ""
echo "Step 4: Comparing outputs..."
python -c "
import pandas as pd
import numpy as np

# Load both runs
X1 = pd.read_csv('$RUN1_DIR/X_features.csv')
X2 = pd.read_csv('$RUN2_DIR/X_features.csv')

y1 = pd.read_csv('$RUN1_DIR/y_target.csv')
y2 = pd.read_csv('$RUN2_DIR/y_target.csv')

# Compare
X_identical = X1.equals(X2)
y_identical = y1.equals(y2)

print(f'X_features identical: {X_identical}')
print(f'y_target identical: {y_identical}')

if X_identical and y_identical:
    print('\n✅ SUCCESS: Outputs are identical - reproducibility verified!')
    exit(0)
else:
    print('\n❌ FAILURE: Outputs differ - reproducibility issue detected!')
    print(f'\nX difference (sum of absolute differences): {(X1 - X2).abs().sum().sum()}')
    print(f'y difference (sum of absolute differences): {(y1 - y2).abs().sum().sum()}')
    exit(1)
"

RESULT=$?

echo ""
echo "Step 5: Cleaning up..."
rm -rf $RUN1_DIR $RUN2_DIR

if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✅ REPRODUCIBILITY TEST PASSED"
    echo -e "==========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "❌ REPRODUCIBILITY TEST FAILED"
    echo -e "==========================================${NC}"
    exit 1
fi
