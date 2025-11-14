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
│   └── load_env.sh      # Load environment variables
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