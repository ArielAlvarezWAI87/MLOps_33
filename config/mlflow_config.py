"""MLflow configuration"""
import mlflow
from pathlib import Path

class MLflowConfig:
    """MLflow configuration manager"""
    
    def __init__(self, tracking_uri="file:./mlruns", experiment_name="energy-consumption-forecasting"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
    def setup(self):
        """Setup MLflow"""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if doesn't exist
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(self.experiment_name)
        
        print(f"✅ MLflow configured:")
        print(f"   Tracking URI: {self.tracking_uri}")
        print(f"   Experiment: {self.experiment_name}")
        
    @staticmethod
    def log_model_info(model, model_name, signature=None):
        """Log model with metadata"""
        mlflow.sklearn.log_model(
            model,
            model_name,
            signature=signature,
            registered_model_name=model_name
        )