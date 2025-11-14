"""
test_api.py
───────────────────────────────────────────────
Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


@pytest.fixture
def api_client(temp_model_dir, temp_data_dir, sample_features_and_target, mocker):
    """Create a test client with mocked model loading"""
    from src.deployment.api import app
    import joblib
    from sklearn.preprocessing import StandardScaler
    from imodels import RuleFitRegressor

    # Create and save a simple model
    X, y = sample_features_and_target
    model = RuleFitRegressor(max_rules=5, random_state=42)
    model.fit(X, y)

    # Create a simple preprocessor
    scaler = StandardScaler()
    scaler.fit(X)

    # Save to temp directories
    model_path = temp_model_dir / 'rulefit.pkl'
    preprocessor_path = temp_data_dir['processed'] / 'preprocessor.pkl'

    joblib.dump(model, model_path)
    joblib.dump(scaler, preprocessor_path)

    # Mock the paths in the API
    mocker.patch('src.deployment.api.MODEL_PATH', model_path)
    mocker.patch('src.deployment.api.PREPROCESSOR_PATH', preprocessor_path)

    # Create test client
    client = TestClient(app)

    # Trigger startup event
    with client:
        yield client


@pytest.mark.unit
class TestAPIEndpoints:
    """Tests for API endpoints"""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info"""
        response = api_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
        assert "timestamp" in data

    def test_model_info_endpoint(self, api_client):
        """Test model info endpoint"""
        response = api_client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "model_type" in data
        assert data["model_type"] == "RuleFitRegressor"

    def test_predict_endpoint_valid_input(self, api_client):
        """Test prediction endpoint with valid input"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 45.2,
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "Medium_Load"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "model_version" in data
        assert "prediction_timestamp" in data
        assert "n_predictions" in data
        assert len(data["predictions"]) == 1
        assert data["n_predictions"] == 1

    def test_predict_endpoint_multiple_inputs(self, api_client):
        """Test prediction endpoint with multiple inputs"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 45.2,
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "Medium_Load"
                },
                {
                    "date": "2018-01-15 15:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 50.0,
                    "Leading_Current_Reactive_Power_kVarh": 15.0,
                    "CO2(tCO2)": 0.010,
                    "Lagging_Current_Power_Factor": 0.90,
                    "Leading_Current_Power_Factor": 0.82,
                    "NSM": 6000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "Maximum_Load"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 2
        assert data["n_predictions"] == 2

    def test_predict_endpoint_invalid_week_status(self, api_client):
        """Test prediction endpoint with invalid WeekStatus"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 45.2,
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "InvalidStatus",
                    "Day_of_week": "Monday",
                    "Load_Type": "Medium_Load"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_load_type(self, api_client):
        """Test prediction endpoint with invalid Load_Type"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 45.2,
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "InvalidLoad"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_out_of_range_values(self, api_client):
        """Test prediction endpoint with out-of-range values"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    "Lagging_Current_Reactive_Power_kVarh": 150.0,  # Too high
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "Medium_Load"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self, api_client):
        """Test prediction endpoint with missing required field"""
        request_data = {
            "data": [
                {
                    "date": "2018-01-15 14:30:00",
                    # Missing Lagging_Current_Reactive_Power_kVarh
                    "Leading_Current_Reactive_Power_kVarh": 12.5,
                    "CO2(tCO2)": 0.008,
                    "Lagging_Current_Power_Factor": 0.85,
                    "Leading_Current_Power_Factor": 0.78,
                    "NSM": 5000,
                    "WeekStatus": "Weekday",
                    "Day_of_week": "Monday",
                    "Load_Type": "Medium_Load"
                }
            ]
        }

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_empty_data(self, api_client):
        """Test prediction endpoint with empty data array"""
        request_data = {"data": []}

        response = api_client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_openapi_schema(self, api_client):
        """Test that OpenAPI schema is accessible"""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
