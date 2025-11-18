# MLOps Project - Steel Energy Prediction

Machine learning project for predicting energy consumption in steel manufacturing with complete MLOps implementation.

## Quick Start

### Setup (New Team Members)

```bash
# Clone and setup
git clone <your-repo-url>
cd <repo-name>
./scripts/setup.sh
```

The script will:
- Create `.env` from template (add your AWS credentials)
- Create virtual environment
- Install dependencies
- Configure DVC and pull data from S3

**Required AWS Credentials** (add to `.env`):
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=abc123...
AWS_DEFAULT_REGION=us-east-2
AWS_PROFILE=equipo0

DVC_REMOTE_NAME=team_remote
DVC_S3_BUCKET=itesm-mna
DVC_S3_PATH=202502-equipo0
```

### Daily Workflow

```bash
# Activate and pull latest 
source .venv/bin/activate #MAC/LINUX
source .venv/Scripts/activate #WINDOWS
source scripts/load_env.sh
dvc pull

# Work and push changes
dvc add data/processed models/
dvc push
git add . && git commit -m "Update model" && git push
```

## Model Serving API

### Start API

```bash
./scripts/start_api.sh
# Or manually: uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Model Information

- **Model**: RuleFitRegressor (`rulefit_steel_energy` v1.0.0)
- **Registry**: `models:/rulefit_steel_energy/1.0.0`
- **Files**: `models/rulefit.pkl`, `data/processed/preprocessor.pkl`

### API Endpoints

#### GET / - API Information
```bash
curl http://localhost:8000/
```

#### GET /health - Health Check
```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true, "preprocessor_loaded": true}
```

#### GET /model/info - Model Metadata
```bash
curl http://localhost:8000/model/info
# {"model_name": "rulefit_steel_energy", "model_version": "1.0.0", ...}
```

#### POST /predict - Make Predictions

**Request:**
```bash
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

**Response:**
```json
{
  "predictions": [75.3],
  "model_version": "1.0.0",
  "prediction_timestamp": "2025-01-15T10:30:00",
  "n_predictions": 1
}
```

**Validation Rules:**
- `Lagging_Current_Reactive_Power_kVarh`: 0-100
- `Leading_Current_Reactive_Power_kVarh`: 0-30
- `CO2(tCO2)`: 0-0.02
- Power Factors: 0-1
- `NSM`: 0-90000
- `WeekStatus`: "Weekday" or "Weekend"
- `Load_Type`: "Light_Load", "Medium_Load", or "Maximum_Load"

### Testing API

```bash
# Quick test
./examples/test_api.sh

# Or manually
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/api_request_example.json
```

## Testing

### Run Tests

```bash
# Quick test
pytest -q

# Verbose with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test types
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only
```

### Test Coverage

**Current: 69% coverage, 30 tests passing**

- `src/data/load_data.py`: 100%
- `src/data/preprocessing.py`: 91%
- `src/features/feature_engineering.py`: 95%
- `src/models/rulefit_trainer.py`: 88%

**Structure:**
```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # 26 tests
│   ├── test_data.py     # Data loading & preprocessing
│   ├── test_features.py # Feature engineering
│   └── test_model.py    # Model training & prediction
└── integration/          # 4 tests
    ├── test_pipeline.py # End-to-end pipeline
    └── test_api.py      # API integration tests
```

## Reproducibility & Docker

### Quick Reproducibility Check

```bash
# Local test
pytest tests/ -v

# Docker test (clean environment)
docker-compose run --rm verify-reproducibility
```

### What Makes It Reproducible

1. **Fixed Dependencies**
   - Python 3.11.3 (`.python-version`)
   - Pinned: `requirements.txt` (17 packages)
   - Complete freeze: `requirements-lock.txt` (178 packages)

2. **Random Seed Management**
   - Global seed: `RANDOM_SEED = 42`
   - Centralized in `src/utils/reproducibility.py`
   - Applied to: NumPy, scikit-learn, RuleFit, train-test splits

3. **Version Control**
   - DVC for data/models (S3: `s3://itesm-mna/202502-equipo0`)
   - MLflow for experiment tracking
   - Docker for environment isolation

### Docker Usage

**Build and run:**
```bash
# Build
docker build -t ml-service:latest .

# Run API
docker run -p 8000:8000 ml-service:latest

# Run in background
docker run -d -p 8000:8000 --name ml-service ml-service:latest

# View logs
docker logs -f ml-service
```

**Docker Compose:**
```bash
# Run tests
docker-compose run --rm test

# Verify reproducibility
docker-compose run --rm verify-reproducibility

# Run training pipeline
docker-compose run --rm train

# Start API service
docker-compose up api
```

**Publish to DockerHub:**
```bash
export DOCKERHUB_USERNAME=your-username
./scripts/docker_publish.sh
```

Tags: `latest`, `1.0.0`, `1.0`, `1`

**Image Details:**
- Base: `python:3.11.3-slim`
- Size: ~1.94GB
- Port: 8000
- Health check: `/health` endpoint (every 30s)

### Cross-Machine Verification

**Machine A:**
```bash
python src/data/preprocessing.py
python src/features/feature_engineering.py
python src/models/rulefit_trainer.py
dvc push && git push
```

**Machine B:**
```bash
git pull && dvc pull
python src/data/preprocessing.py
python src/features/feature_engineering.py
python src/models/rulefit_trainer.py
# Results identical to Machine A
```

## Model Evaluation

Evaluate model performance with baseline and drift simulation:

```bash
python src/evaluation/evaluate.py
```

This will:
- Run baseline evaluation on processed data
- Simulate data drift (25% feature shift)
- Compare performance metrics
- Log results to MLflow

**View results in MLflow UI:**
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open: http://localhost:5000
```

**Example output:**
```
📊 BASELINE EVALUATION
✓ Baseline metrics computed
  MAE:  4.2230 kWh
  RMSE: 7.0438 kWh
  R²:   0.9556

🌪 DRIFT EVALUATION
✓ Drift metrics computed
  Baseline - RMSE: 7.0438 kWh
  Drifted  - RMSE: 10.4366 kWh
  Degradation: 48.17%
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Original data (DVC tracked)
│   └── processed/        # Processed data (DVC tracked)
├── models/               # Trained models (DVC tracked)
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training
│   ├── evaluation/      # Model evaluation and drift simulation
│   ├── deployment/      # FastAPI service
│   └── utils/           # Reproducibility utilities
├── tests/                # 30 tests, 69% coverage
│   ├── unit/            # 26 unit tests
│   └── integration/     # 4 integration tests
├── scripts/
│   ├── setup.sh              # One-command setup
│   ├── load_env.sh           # Load environment variables
│   ├── start_api.sh          # Start FastAPI server
│   ├── docker_publish.sh     # Publish to DockerHub
│   └── compare_outputs.py    # Compare pipeline outputs (for Docker reproducibility test)
├── examples/             # API testing examples
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service orchestration
├── pytest.ini           # Test configuration
├── requirements.txt     # Main dependencies (pinned)
├── requirements-lock.txt # Complete freeze
└── .python-version      # Python 3.11.3
```

## Troubleshooting

### DVC Issues

**"Access Denied" on dvc pull:**
```bash
# Verify credentials
cat .env
aws s3 ls s3://itesm-mna/202502-equipo0 --profile equipo0
```

**"No remote configured":**
```bash
./scripts/setup.sh  # Reconfigure DVC
```

### Environment Issues

**Reset everything:**
```bash
rm -rf .venv .dvc
./scripts/setup.sh
```

**Import errors when running scripts directly:**
```bash
# Use these commands from project root
python src/data/preprocessing.py
python src/features/feature_engineering.py
python src/models/rulefit_trainer.py
```

### Docker Issues

**Container won't start:**
```bash
docker logs ml-service  # Check logs
docker ps -a           # Check status
```

**Port already in use:**
```bash
# Use different port
docker run -p 9000:8000 ml-service:latest
```

## MLOps Features Summary

✅ **Testing**: 30 unit & integration tests, 69% coverage
✅ **API Serving**: FastAPI with Pydantic validation, OpenAPI docs
✅ **Reproducibility**: Fixed deps, random seeds, Docker, DVC
✅ **Containerization**: Docker images with semantic versioning
✅ **Model Evaluation**: Baseline and drift simulation with MLflow tracking

---

**Model**: RuleFitRegressor | **Framework**: scikit-learn, MLflow | **Deployment**: FastAPI, Docker
