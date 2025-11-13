#!/bin/bash
set -e  # Exit on error

echo "🚀 Setting up MLOps project..."
echo ""

# Step 1: Check for .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your AWS credentials"
    echo ""
    echo "Required credentials:"
    echo "  - AWS_ACCESS_KEY_ID"
    echo "  - AWS_SECRET_ACCESS_KEY"
    echo "  - AWS_PROFILE (if using AWS CLI profiles)"
    echo ""
    echo "After editing .env, run this script again:"
    echo "  ./scripts/setup.sh"
    exit 1
fi

# Step 2: Load environment variables
echo "📋 Loading environment variables from .env..."
export $(cat .env | grep -v '^#' | xargs)

# Validate required variables
if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "❌ Error: AWS credentials not found in .env"
    echo "Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env"
    exit 1
fi

if [ -z "${DVC_REMOTE_NAME}" ] || [ -z "${DVC_S3_BUCKET}" ] || [ -z "${DVC_S3_PATH}" ]; then
    echo "❌ Error: DVC configuration incomplete in .env"
    echo "Please ensure DVC_REMOTE_NAME, DVC_S3_BUCKET, and DVC_S3_PATH are set"
    exit 1
fi

echo "✓ Environment variables loaded"
echo ""

# Step 3: Create virtual environment if needed
if [ ! -d .venv ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 4: Activate virtual environment
echo "🔌 Activating virtual environment..."
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    echo "❌ Error: Could not find activation script"
    exit 1
fi
echo "✓ Virtual environment activated"
echo ""

# Step 5: Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 6: Initialize DVC if needed
if [ ! -d .dvc ]; then
    echo "📦 Initializing DVC..."
    dvc init
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi
echo ""

# Step 7: Configure DVC remote
echo "🔗 Configuring DVC remote..."

# Remove existing remote if it exists
if dvc remote list | grep -q "${DVC_REMOTE_NAME}"; then
    echo "  Removing existing remote: ${DVC_REMOTE_NAME}"
    dvc remote remove ${DVC_REMOTE_NAME}
fi

# Add remote
echo "  Adding remote: ${DVC_REMOTE_NAME}"
dvc remote add -d ${DVC_REMOTE_NAME} s3://${DVC_S3_BUCKET}/${DVC_S3_PATH}

# Set region
echo "  Setting region: ${AWS_DEFAULT_REGION}"
dvc remote modify ${DVC_REMOTE_NAME} region ${AWS_DEFAULT_REGION}

# Set profile if specified
if [ ! -z "${AWS_PROFILE}" ]; then
    echo "  Setting AWS profile: ${AWS_PROFILE}"
    dvc remote modify ${DVC_REMOTE_NAME} profile ${AWS_PROFILE}
fi

echo "✓ DVC remote configured"
echo ""

# Step 8: Show configuration summary
echo "📊 Configuration Summary:"
echo "  DVC Remote: ${DVC_REMOTE_NAME}"
echo "  S3 URL: s3://${DVC_S3_BUCKET}/${DVC_S3_PATH}"
echo "  Region: ${AWS_DEFAULT_REGION}"
echo "  Profile: ${AWS_PROFILE:-<not set>}"
echo ""

# Step 9: Test DVC connection and pull data
echo "📥 Testing DVC connection and pulling data..."
if dvc pull; then
    echo "✓ Data pulled successfully"
else
    echo "⚠️  Warning: Could not pull data. This might be normal if:"
    echo "  - No data has been pushed yet"
    echo "  - Your AWS credentials need verification"
    echo "  - You don't have access to the S3 bucket yet"
fi
echo ""

# Step 10: Final instructions
echo "✅ Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Next Steps:"
echo ""
echo "1. Commit DVC configuration (first time only):"
echo "   git add .dvc/config .dvc/.gitignore .dvcignore"
echo "   git commit -m 'Configure DVC remote'"
echo ""
echo "2. Start working:"
echo "   source .venv/bin/activate"
echo "   source scripts/load_env.sh"
echo ""
echo "3. Daily workflow:"
echo "   dvc pull   # Get latest data"
echo "   # ... do your work ..."
echo "   dvc push   # Upload data changes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"