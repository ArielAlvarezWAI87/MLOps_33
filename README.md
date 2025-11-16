# MLOps Project

## Quick Start (New Team Members)

### One-Command Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Run the setup script
./scripts/setup.sh
```

That's it! The script will:
- Create `.env` from template (you'll need to add credentials)
- Create virtual environment
- Install all dependencies
- Initialize and configure DVC
- Pull data from S3

**First time running?** The script will create `.env` and exit. Edit it with your credentials, then run `./scripts/setup.sh` again.

### What You Need

**AWS Credentials** (get these from your team lead):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- Optionally: `AWS_PROFILE` if using AWS CLI profiles

**Add them to `.env`:**
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=abc123...
AWS_DEFAULT_REGION=us-east-2
AWS_PROFILE=equipo0

DVC_REMOTE_NAME=team_remote
DVC_S3_BUCKET=itesm-mna
DVC_S3_PATH=202502-equipo0
```

## Daily Workflow
```bash
# Activate environment and load credentials
source .venv/bin/activate
source scripts/load_env.sh

# Pull latest data
dvc pull

# Do your work
python src/train.py

# Push any data/model changes
dvc add data/processed models/
dvc push

# Commit and push code changes
git add .
git commit -m "Update model"
git push
```

## Model Serving API

### Starting the API Server

**Quick start:**
```bash
./scripts/start_api.sh
```

**Manual start:**
```bash
# Activate environment
source .venv/bin/activate

# Start API server
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Model Information

**Current Model:**
- **Name**: `rulefit_steel_energy`
- **Version**: `1.0.0`
- **Type**: RuleFitRegressor
- **Model Path**: `models/rulefit.pkl`
- **Preprocessor Path**: `data/processed/preprocessor.pkl`

**Model Registry (MLflow):**
```
models:/rulefit_steel_energy/1.0.0
```

### API Endpoints

#### 1. **GET /** - API Information
Get basic information about the API.

```bash
curl http://localhost:8000/
```

#### 2. **GET /health** - Health Check
Check if the API and model are ready.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "timestamp": "2025-01-15T10:30:00"
}
```

#### 3. **GET /model/info** - Model Metadata
Get information about the loaded model.

```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "model_name": "rulefit_steel_energy",
  "model_version": "1.0.0",
  "model_path": "models/rulefit.pkl",
  "preprocessor_path": "data/processed/preprocessor.pkl",
  "model_type": "RuleFitRegressor"
}
```

#### 4. **POST /predict** - Make Predictions
Predict energy consumption for steel manufacturing.

**Request Schema:**
```json
{
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
```

**Field Validations:**
- `Lagging_Current_Reactive_Power_kVarh`: 0 ≤ value ≤ 100
- `Leading_Current_Reactive_Power_kVarh`: 0 ≤ value ≤ 30
- `CO2(tCO2)`: 0 ≤ value ≤ 0.02
- `Lagging_Current_Power_Factor`: 0 ≤ value ≤ 1
- `Leading_Current_Power_Factor`: 0 ≤ value ≤ 1
- `NSM`: 0 ≤ value ≤ 90000
- `WeekStatus`: Must be "Weekday" or "Weekend"
- `Day_of_week`: Must be valid day name
- `Load_Type`: Must be "Light_Load", "Medium_Load", or "Maximum_Load"

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/api_request_example.json
```

**Response:**
```json
{
  "predictions": [75.3, 82.1],
  "model_version": "1.0.0",
  "prediction_timestamp": "2025-01-15T10:30:00",
  "n_predictions": 2
}
```

### Testing the API

**Automated test script:**
```bash
./examples/test_api.sh
```

**Manual testing with curl:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
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
    }]
  }'
```

**Using Python:**
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": [{
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
        }]
    }
)

result = response.json()
print(f"Predicted energy: {result['predictions'][0]} kWh")
```

### Error Handling

The API provides detailed error messages for:
- **400 Bad Request**: Invalid input data, validation errors
- **422 Unprocessable Entity**: Pydantic validation failures
- **500 Internal Server Error**: Model prediction errors
- **503 Service Unavailable**: Model not loaded

**Example Error Response:**
```json
{
  "error": "Invalid input data: WeekStatus must be one of ['Weekday', 'Weekend']",
  "detail": "ValidationError...",
  "timestamp": "2025-01-15T10:30:00"
}
```

### Deployment Considerations

**Production deployment:**
- Use `gunicorn` or multiple `uvicorn` workers for production
- Set up HTTPS/TLS certificates
- Implement rate limiting
- Add authentication/authorization
- Use containerization (Docker)
- Set up monitoring and logging

**Example production command:**
```bash
gunicorn src.deployment.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Testing

### Running Tests

**Quick command (as specified):**
```bash
pytest -q
```

**Other useful commands:**
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Then open htmlcov/index.html in your browser

# Run specific test file
pytest tests/unit/test_data.py

# Run tests matching a pattern
pytest tests/ -k "preprocessing"
```

### Test Coverage

Current coverage: **62%**
- `src/data/load_data.py`: 100%
- `src/data/preprocessing.py`: 91%
- `src/features/feature_engineering.py`: 95%
- `src/models/rulefit_trainer.py`: 88%

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and utilities
├── unit/                    # Unit tests (17 tests)
│   ├── test_data.py        # Data loading & preprocessing (11 tests)
│   ├── test_features.py    # Feature engineering (3 tests)
│   └── test_model.py       # Model training & prediction (5 tests)
└── integration/             # Integration tests (2 tests)
    └── test_pipeline.py    # End-to-end pipeline validation
```

### What is Tested

**Unit Tests:**
- ✅ Data loading and validation
- ✅ Data preprocessing and cleaning
- ✅ Data quality rules (power factor, CO2, NSM limits)
- ✅ Missing value handling
- ✅ Feature engineering (temporal, cyclical, power features)
- ✅ Model training with RuleFit
- ✅ Model serialization/deserialization
- ✅ Prediction workflows

**Integration Tests:**
- ✅ End-to-end pipeline: data → preprocessing → features → model
- ✅ Data flow consistency across pipeline stages

### Adding New Tests

1. **For unit tests**, add to appropriate file in `tests/unit/`:
```python
@pytest.mark.unit
def test_my_new_feature(sample_data):
    # Your test here
    assert result == expected
```

2. **For integration tests**, add to `tests/integration/test_pipeline.py`:
```python
@pytest.mark.integration
def test_new_pipeline_flow(temp_data_dir, sample_raw_data):
    # Your test here
    assert pipeline_works
```

3. **Use fixtures** from `tests/conftest.py`:
- `sample_raw_data` - Mock raw dataset
- `sample_processed_data` - Mock processed dataset
- `temp_data_dir` - Temporary directories for testing
- `mock_mlflow_run` - Mock MLflow tracking

## Project Structure
```
.
├── data/
│   ├── raw/              # Original data (DVC tracked)
│   └── processed/        # Processed data (DVC tracked)
├── models/               # Trained models (DVC tracked)
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   ├── evaluation/      # Model evaluation
│   └── deployment/      # API and deployment
├── tests/                # Test suite (19 tests, 62% coverage)
│   ├── conftest.py      # Shared test fixtures
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── scripts/
│   ├── setup.sh         # One-command setup
│   ├── load_env.sh      # Load environment variables
│   └── start_api.sh     # Start FastAPI server
├── examples/             # Example files for API testing
│   ├── api_request_example.json  # Sample API request
│   └── test_api.sh      # API testing script
├── pytest.ini           # Pytest configuration
├── .env.example         # Template for credentials
├── .env                 # Your credentials (not committed)
├── requirements.txt     # Python dependencies
└── README.md
```

## Troubleshooting

### "Access Denied" when running dvc pull
- Verify your AWS credentials in `.env`
- Ask your team lead to grant you S3 bucket access
- Test with: `aws s3 ls s3://itesm-mna/202502-equipo0 --profile equipo0`

### "No remote configured"
- Run: `./scripts/setup.sh` again
- This will reconfigure the DVC remote

### Need to reset everything?
```bash
# Remove virtual environment and DVC
rm -rf .venv .dvc

# Run setup again
./scripts/setup.sh
```

## Model Evaluation and Data Drift Monitoring

### Offline model evaluation

Once the project is set up (`./scripts/setup.sh`, environment activated and `dvc pull` completed), you can run an offline evaluation of the current model:

```bash
python -m src.evaluation.evaluate