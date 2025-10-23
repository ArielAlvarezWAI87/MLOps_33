"""Data loading utilities"""
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(data_path: str = 'data/raw/steel_energy_modified.csv') -> pd.DataFrame:
    """Load raw data from CSV"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded {len(df)} records from {data_path}")
        return df
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {data_path}")
        raise

def load_processed_data(data_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed features and target"""
    X = pd.read_csv(f"{data_dir}/X_features.csv")
    y = pd.read_csv(f"{data_dir}/y_target.csv").values.ravel()
    
    logger.info(f"✅ Loaded processed data: X{X.shape}, y{y.shape}")
    return X, y