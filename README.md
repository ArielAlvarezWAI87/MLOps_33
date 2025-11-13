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

## Project Structure
```
.
├── data/
│   ├── raw/              # Original data (DVC tracked)
│   └── processed/        # Processed data (DVC tracked)
├── models/               # Trained models (DVC tracked)
├── src/                  # Source code
├── scripts/
│   ├── setup.sh         # One-command setup
│   └── load_env.sh      # Load environment variables
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