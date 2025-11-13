#!/bin/bash
# Load environment variables for DVC
# Usage: source scripts/load_env.sh

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded"
    echo "  Region: ${AWS_DEFAULT_REGION}"
    echo "  Profile: ${AWS_PROFILE:-<not set>}"
else
    echo "❌ .env file not found"
    echo "Run: cp .env.example .env and add your credentials"
    exit 1
fi