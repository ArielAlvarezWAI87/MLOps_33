"""Feature engineering module"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        df_eng = df.copy()
        
        # Tenure categories
        df_eng['tenure_category'] = pd.cut(
            df_eng['tenure_months'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12', '12-24', '24-48', '48+']
        )
        
        # Average monthly spend
        df_eng['avg_monthly_spend'] = df_eng['total_charges'] / (df_eng['tenure_months'] + 1)
        
        # Service score
        service_cols = ['has_phone_service', 'has_internet', 'has_tech_support']
        df_eng['service_score'] = df_eng[service_cols].sum(axis=1)
        
        # High value customer
        df_eng['is_high_value'] = (df_eng['monthly_charges'] > df_eng['monthly_charges'].median()).astype(int)
        
        # Contract stability
        contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
        df_eng['contract_stability'] = df_eng['contract_type'].map(contract_map)
        
        logger.info(f"✅ Engineered {len(df_eng.columns)} features")
        return df_eng
    
    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Label encoding for contract type
        if fit:
            self.label_encoders['contract_type'] = LabelEncoder()
            df_encoded['contract_type_encoded'] = self.label_encoders['contract_type'].fit_transform(df_encoded['contract_type'])
        else:
            df_encoded['contract_type_encoded'] = self.label_encoders['contract_type'].transform(df_encoded['contract_type'])
        
        # One-hot encoding
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=['payment_method', 'tenure_category'],
            prefix=['payment', 'tenure_cat'],
            drop_first=True
        )
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'churn', fit: bool = True):
        """Full feature preparation pipeline"""
        # Engineer features
        df_eng = self.engineer_features(df)
        
        # Encode categoricals
        df_encoded = self.encode_categoricals(df_eng, fit=fit)
        
        # Separate features and target
        feature_cols = [col for col in df_encoded.columns if col not in ['customer_id', target_col, 'contract_type']]
        X = df_encoded[feature_cols]
        y = df_encoded[target_col] if target_col in df_encoded.columns else None
        
        # Store feature names
        if fit:
            self.feature_names = list(X.columns)
        
        logger.info(f"✅ Prepared features: {X.shape}")
        return X, y
    
    def save(self, path: str):
        """Save feature engineer"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"✅ Saved feature engineer to {path}")
    
    @staticmethod
    def load(path: str):
        """Load feature engineer"""
        return joblib.load(path)