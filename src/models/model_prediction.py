"""
model_prediction.py
───────────────────────────────────────────────
Module: RuleFitPredictor
Author: Bruno Sánchez García
Purpose:
    Load a pre-trained RuleFitRegressor and preprocessing
    pipeline to perform predictions on new data.

Input:
    ../data/processed/preprocessor.pkl
    ../models/rulefit.pkl

Output:
    Predictions (numpy.ndarray)
"""

import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import sys
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════
# MLflow Configuration
# ══════════════════════════════════════════════════════════════════════════
# mlflow.set_tracking_uri("file:../mlruns")
# mlflow.set_experiment("rulefit_predictions")


class RuleFitPredictor:
    """
    Loads a trained RuleFitRegressor model and its preprocessing pipeline,
    then performs predictions on new input data.
    """

    def __init__(self,
                 model_path: str,
                 preprocessor_path: str):
        """
        Initialize predictor with paths to model and preprocessor.

        Args:
            model_path (str): Path to the trained RuleFit model (.pkl)
            preprocessor_path (str): Path to the saved preprocessing pipeline (.pkl)
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None

    # ----------------------------------------------------------------------
    def load_model(self):
        """Load the serialized RuleFitRegressor model."""
        print(f"\n📦 Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print("✅ Model loaded successfully!")

    # ----------------------------------------------------------------------
    def load_model_from_mlflow(self, model_name: str, model_version: str = None,
                               model_stage: str = None):
        """
        Load model from MLflow Model Registry.

        Args:
            model_name (str): Name of the registered model in MLflow
            model_version (str): Specific version to load (e.g., "1", "2")
            model_stage (str): Stage to load from (e.g., "Production", "Staging")
                             If both version and stage are None, loads latest version

        Example:
            predictor.load_model_from_mlflow("rulefit_model", model_stage="Production")
        """
        try:
            if model_version:
                model_uri = f"models:/{model_name}/{model_version}"
                print(
                    f"\n📦 Loading model from MLflow Registry: {model_name} (v{model_version})")
            elif model_stage:
                model_uri = f"models:/{model_name}/{model_stage}"
                print(
                    f"\n📦 Loading model from MLflow Registry: {model_name} ({model_stage})")
            else:
                model_uri = f"models:/{model_name}/latest"
                print(
                    f"\n📦 Loading latest model from MLflow Registry: {model_name}")

            self.model = mlflow.sklearn.load_model(model_uri)
            print("✅ Model loaded successfully from MLflow!")
        except Exception as e:
            print(f"❌ Failed to load model from MLflow: {e}")
            print("   Falling back to loading from file path...")
            self.load_model()

    # ----------------------------------------------------------------------

    def load_preprocessor(self):
        """Load the preprocessing pipeline."""
        print(f"\n⚙️  Loading preprocessor from: {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)
        print("✅ Preprocessor loaded successfully!")

    # ----------------------------------------------------------------------
    def preprocess(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to raw input data.

        Args:
            X_raw (pd.DataFrame): New raw data (must match training schema)
        Returns:
            X_transformed (pd.DataFrame): Scaled and encoded features
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not loaded. Call load_preprocessor() first.")

        print("\n🔄 Applying preprocessing transformations...")
        X_transformed = self.preprocessor.transform(X_raw)
        print(f"✅ Data transformed: {X_transformed.shape}")
        return X_transformed

    # ----------------------------------------------------------------------
    def predict(self, X_raw: pd.DataFrame, log_to_mlflow: bool = True,
                run_name: str = None, log_model: bool = True,
                register_model: str = None) -> np.ndarray:
        """
        Perform full prediction pipeline (preprocess → model.predict).

        Args:
            X_raw (pd.DataFrame): New raw input data.
            log_to_mlflow (bool): Whether to log predictions to MLflow.
            run_name (str): Optional name for the MLflow run.
            log_model (bool): Whether to log the model artifact to MLflow.
            register_model (str): If provided, register model with this name in Model Registry.
                                 Example: "rulefit_energy_model"
        Returns:
            np.ndarray: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not loaded. Call load_preprocessor() first.")

        print("\n🚀 Starting prediction pipeline...")

        if log_to_mlflow:
            # Start MLflow run for prediction tracking
            run_name = run_name or f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                mlflow.log_param("model_path", str(self.model_path))
                mlflow.log_param("preprocessor_path",
                                 str(self.preprocessor_path))
                mlflow.log_param("input_shape", X_raw.shape)
                mlflow.log_param("n_samples", X_raw.shape[0])
                mlflow.log_param("n_features", X_raw.shape[1])
                mlflow.log_param("model_logged", log_model)
                mlflow.log_param("model_registered", bool(register_model))

                # Perform prediction
                X_processed = self.preprocess(X_raw)
                y_pred = self.model.predict(X_processed)

                # Log prediction statistics as metrics
                mlflow.log_metric("predictions_mean", float(np.mean(y_pred)))
                mlflow.log_metric("predictions_std", float(np.std(y_pred)))
                mlflow.log_metric("predictions_min", float(np.min(y_pred)))
                mlflow.log_metric("predictions_max", float(np.max(y_pred)))
                mlflow.log_metric("predictions_median",
                                  float(np.median(y_pred)))

                # Log input data statistics
                mlflow.log_metric("input_data_completeness",
                                  float(1 - X_raw.isnull().sum().sum() / (X_raw.shape[0] * X_raw.shape[1])))

                # Save predictions as artifact
                predictions_df = pd.DataFrame({
                    'prediction': y_pred,
                    'timestamp': datetime.now()
                })
                predictions_path = "predictions_output.csv"
                predictions_df.to_csv(predictions_path, index=False)
                mlflow.log_artifact(predictions_path)

                # Log model info tags
                mlflow.set_tag("model_type", "RuleFitRegressor")
                mlflow.set_tag("prediction_type", "batch")
                mlflow.set_tag("mlflow.runName", run_name)

                # ══════════════════════════════════════════════════════════════
                # LOG MODEL TO MLFLOW (appears in Models tab)
                # ══════════════════════════════════════════════════════════════
                if log_model:
                    try:
                        print("\n📦 Logging model to MLflow...")

                        # Infer model signature (input/output schema)
                        signature = infer_signature(X_processed, y_pred)

                        # Create input example for model documentation
                        # First 5 rows as example
                        input_example = X_processed[:5]

                        # Log the model with signature and example
                        mlflow.sklearn.log_model(
                            sk_model=self.model,
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example,
                            registered_model_name=register_model  # This registers it!
                        )

                        print(f"✅ Model logged to MLflow successfully!")

                        if register_model:
                            print(f"✅ Model registered as: '{register_model}'")
                            print(f"   Check Models tab at http://localhost:5000")
                        else:
                            print(
                                f"   Model saved in run artifacts (not registered)")
                            print(
                                f"   To register, use: register_model='your_model_name'")

                    except Exception as e:
                        print(f"⚠️  Warning: Could not log model: {e}")
                        print(f"   Prediction will continue normally...")

                # Log run ID for reference
                print(f"\n✅ Prediction logged to MLflow")
                print(f"   Run Name: {run_name}")
                print(f"   Run ID: {run.info.run_id}")

        else:
            # Run without MLflow tracking
            X_processed = self.preprocess(X_raw)
            y_pred = self.model.predict(X_processed)

        print("✅ Prediction completed successfully!")
        return y_pred
    # ----------------------------------------------------------------------

    def register_current_model(self, model_name: str, run_id: str = None,
                               description: str = None, tags: dict = None):
        """
        Register the current model in MLflow Model Registry.

        Args:
            model_name (str): Name to register the model under
            run_id (str): MLflow run ID containing the model (if None, uses latest run)
            description (str): Optional description of the model
            tags (dict): Optional tags for the model version

        Returns:
            ModelVersion object from MLflow

        Example:
            predictor.register_current_model(
                model_name="rulefit_energy_predictor",
                description="Production model for energy prediction",
                tags={"environment": "production", "version": "2.1.0"}
            )
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        try:
            # If no run_id provided, get the latest run
            if run_id is None:
                experiment = client.get_experiment_by_name(
                    "rulefit_predictions")
                runs = client.search_runs(
                    experiment.experiment_id,
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs:
                    raise ValueError(
                        "No runs found. Run predict() first with log_to_mlflow=True and log_model=True")
                run_id = runs[0].info.run_id

            # Build model URI
            model_uri = f"runs:/{run_id}/model"

            print(f"\n🏷️  Registering model in Model Registry...")
            print(f"   Model Name: {model_name}")
            print(f"   Run ID: {run_id}")

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            # Add description if provided
            if description:
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )

            print(f"✅ Model registered successfully!")
            print(f"   Model Name: {model_name}")
            print(f"   Version: {model_version.version}")
            print(
                f"   🔗 View in Models tab: http://localhost:5000/#/models/{model_name}")

            return model_version

        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return None

    # ----------------------------------------------------------------------
    def transition_model_stage(self, model_name: str, version: str,
                               stage: str, archive_existing: bool = True):
        """
        Transition a model version to a different stage in Model Registry.

        Args:
            model_name (str): Name of the registered model
            version (str): Version number to transition (e.g., "1", "2")
            stage (str): Target stage - "Staging", "Production", or "Archived"
            archive_existing (bool): Whether to archive existing versions in target stage

        Example:
            # Promote version 3 to Production
            predictor.transition_model_stage("rulefit_energy_predictor", "3", "Production")
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        try:
            print(f"\n🔄 Transitioning model stage...")
            print(f"   Model: {model_name}")
            print(f"   Version: {version}")
            print(f"   New Stage: {stage}")

            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )

            print(f"✅ Model stage updated successfully!")
            print(
                f"   🔗 View: http://localhost:5000/#/models/{model_name}/versions/{version}")

        except Exception as e:
            print(f"❌ Failed to transition model stage: {e}")


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")))
    from src.features.feature_engineering import FeatureEngineer
    # root="C:/Users/AABDC5/repo/PythonML/dev/mlops-homework/"
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MODEL_DIR = ROOT_DIR / "models"
    TMP_DIR = ROOT_DIR / "data" / "tmp"

    MLRUN_DIR = ROOT_DIR / "mlruns"
    mlflow.set_tracking_uri(MLRUN_DIR.as_uri())
    mlflow.set_experiment("rulefit_predictions")

    # Define absolute paths
    model_path = MODEL_DIR / "rulefit.pkl"
    preprocessor_path = PROCESSED_DIR / "preprocessor.pkl"
    data_path = PROCESSED_DIR / "steel_energy_processed.csv"

    # ----------------------------------------------------------------------
    # 1️⃣ Load the raw/cleaned data
    # ----------------------------------------------------------------------
    df_new = pd.read_csv(data_path, parse_dates=["date"])
    print(
        f"\n📂 Loaded new data for inference: {df_new.shape[0]} rows × {df_new.shape[1]} columns")

    # ----------------------------------------------------------------------
    # 2️⃣ Generate engineered features (same as during training)
    # ----------------------------------------------------------------------
    fe = FeatureEngineer(
        raw_processed_dir=PROCESSED_DIR,
        save_dir=TMP_DIR
    )
    df_engineered = fe.engineer_features(df_new)

    # Drop non-feature columns to match training schema
    exclude_cols = ["date", "Usage_kWh", "Day_of_week"]
    X_new = df_engineered[[
        c for c in df_engineered.columns if c not in exclude_cols]]

    print(
        f"✅ Engineered feature matrix ready: {X_new.shape[0]} rows × {X_new.shape[1]} features")

    # ----------------------------------------------------------------------
    # 3️⃣ Initialize and load predictor
    # ----------------------------------------------------------------------
    predictor = RuleFitPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    predictor.load_model()
    predictor.load_preprocessor()

    # ----------------------------------------------------------------------
    # 4️⃣ Run predictions with MLflow tracking and Model Registry
    # ----------------------------------------------------------------------
    run_name = f"steel_energy_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Option 1: Log and register model in one step
    y_pred = predictor.predict(
        X_new,
        log_to_mlflow=True,
        run_name=run_name,
        log_model=True,  # Log model to MLflow
        register_model="rulefit_steel_energy_model"  # Register in Model Registry
    )

    # Option 2: Register model after prediction (if not done above)
    # predictor.register_current_model(
    #     model_name="rulefit_steel_energy_model",
    #     description="RuleFit model for steel plant energy prediction",
    #     tags={"domain": "energy", "plant": "steel_manufacturing"}
    # )

    # Option 3: Transition model to Production stage (after testing in Staging)
    # predictor.transition_model_stage(
    #     model_name="rulefit_steel_energy_model",
    #     version="1",  # Replace with your version number
    #     stage="Production"
    # )

    # ----------------------------------------------------------------------
    # 5️⃣ Display results
    # ----------------------------------------------------------------------
    print("\n📊 Predictions preview:")
    print(y_pred[:10])
    print(f"\nTotal predictions generated: {len(y_pred)}")
    print(f"\n🔗 MLflow UI: mlflow ui --port 5000")
    print(f"   Experiments: http://localhost:5000")
    print(f"   Models Tab: http://localhost:5000/#/models")
    print(f"   Your Model: http://localhost:5000/#/models/rulefit_steel_energy_model")
