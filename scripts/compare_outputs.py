#!/usr/bin/env python3
"""Compare two pipeline runs to verify reproducibility"""

import pandas as pd
import sys

try:
    X1 = pd.read_csv("data/processed/X_features_run1.csv")
    X2 = pd.read_csv("data/processed/X_features_run2.csv")
    y1 = pd.read_csv("data/processed/y_target_run1.csv")
    y2 = pd.read_csv("data/processed/y_target_run2.csv")
    
    X_match = X1.equals(X2)
    y_match = y1.equals(y2)
    
    print(f"X_features match: {X_match}")
    print(f"y_target match: {y_match}")
    
    if X_match and y_match:
        print("\n✅ SUCCESS: Pipeline is reproducible!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Outputs differ!")
        sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
