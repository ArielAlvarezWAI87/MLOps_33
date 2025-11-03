"""
rulefit_trainer.py
───────────────────────────────────────────────
Module: RuleFitTrainer
Author: Bruno Sánchez García
Purpose:
    Train a RuleFitRegressor using processed features and
    save the trained model for later inference.

Inputs:
    ../data/processed/X_features.csv
    ../data/processed/y_target.csv
Outputs:
    ../models/rulefit.pkl
"""

import pandas as pd
import numpy as np
from semver import process
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imodels import RuleFitRegressor
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn


class RuleFitTrainer:
    """Train and export a RuleFitRegressor model."""

    def __init__(self,
                 data_dir: str = "../../data/processed",
                 model_dir: str = "../../models"):
        """
        Initialize directories for data and model saving.

        Args:
            data_dir (str): Folder containing X_features.csv and y_target.csv
            model_dir (str): Destination folder for the trained model
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def load_data(self):
        """Load processed features and target."""
        print("\n📂 Loading processed feature data...")
        X = pd.read_csv(self.data_dir / "X_features.csv")
        y = pd.read_csv(self.data_dir / "y_target.csv").squeeze()
        print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        return X, y

    # ----------------------------------------------------------------------
    def train_rulefit(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      max_rules: int = 30,
                      tree_size: int = 4,
                      random_state: int = 42,
                      exp_rand_tree_size: bool = True,
                      n_jobs=-1):
        """Train a RuleFitRegressor model with standard hyperparameters."""
        print("\n⚙️  Training RuleFitRegressor...")
        mlflow.log_param("max_rules", max_rules)
        mlflow.log_param("tree_size", tree_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("exp_rand_tree_size", exp_rand_tree_size)
        mlflow.log_param("n_jobs", n_jobs)
        
        model = RuleFitRegressor(
            max_rules=max_rules,
            tree_size=tree_size,
            random_state=random_state,
            exp_rand_tree_size=exp_rand_tree_size
        )
        model.fit(X, y)
        print("✅ Model training completed successfully!")
        return model

    # ----------------------------------------------------------------------
    def evaluate(self, model, X, y):
        """Evaluate the trained model using regression metrics."""
        print("\n📈 Evaluating model performance...")
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        return {"r2": r2, "mae": mae, "rmse": rmse}

    # ----------------------------------------------------------------------
    def save_model(self, model, filename: str = "rulefit.pkl"):
        """Save trained model to disk."""
        model_path = self.model_dir / filename
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved successfully to: {model_path}")
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path))
        
        return model_path

    # ----------------------------------------------------------------------
    def run(self):
        """Execute the full training workflow."""
        mlflow.set_experiment("RuleFit_Training")
        
        with mlflow.start_run():
            X, y = self.load_data()
            model = self.train_rulefit(X, y)
            self.evaluate(model, X, y)
            self.save_model(model)
            print("\n✅ RuleFit training pipeline completed successfully!")
            return model


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    #root="C:/Users/BSG/Documents/Academic/2025/MLOps/MLOps_33-main/MLOps_33-main/"
    ROOT_DIR = Path(__file__).resolve().parents[2] 
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MODEL_DIR = ROOT_DIR / "models"

    MLRUN_DIR = ROOT_DIR / "mlruns"
    mlflow.set_tracking_uri(MLRUN_DIR.as_uri())
    mlflow.set_experiment("rulefit_predictions")

    trainer = RuleFitTrainer(
        data_dir = PROCESSED_DIR,
        model_dir = MODEL_DIR
    )
    trainer.run()