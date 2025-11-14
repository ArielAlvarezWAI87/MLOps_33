#!/bin/bash
# start_api.sh
# Start the FastAPI server for model serving

set -e

echo "🚀 Starting Steel Energy Prediction API..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Check if model exists
MODEL_PATH="models/rulefit.pkl"
PREPROCESSOR_PATH="data/processed/preprocessor.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "Please train the model first or pull from DVC: dvc pull"
    exit 1
fi

if [ ! -f "$PREPROCESSOR_PATH" ]; then
    echo "❌ Preprocessor not found at: $PREPROCESSOR_PATH"
    echo "Please run feature engineering or pull from DVC: dvc pull"
    exit 1
fi

echo "✅ Model found: $MODEL_PATH"
echo "✅ Preprocessor found: $PREPROCESSOR_PATH"
echo ""

# Default values
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
RELOAD="${API_RELOAD:-true}"

echo "Starting API server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Reload: $RELOAD"
echo ""
echo "API Documentation will be available at:"
echo "  📚 Swagger UI: http://localhost:$PORT/docs"
echo "  📖 ReDoc:      http://localhost:$PORT/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run uvicorn
if [ "$RELOAD" = "true" ]; then
    uvicorn src.deployment.api:app --host "$HOST" --port "$PORT" --reload
else
    uvicorn src.deployment.api:app --host "$HOST" --port "$PORT"
fi
