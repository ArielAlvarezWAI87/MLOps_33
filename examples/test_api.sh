#!/bin/bash
# test_api.sh
# Script to test the API endpoints

API_URL="${API_URL:-http://localhost:8000}"

echo "Testing Steel Energy Prediction API"
echo "API URL: $API_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint (GET /)..."
curl -s "$API_URL/" | jq '.'
echo ""

# Test 2: Health check
echo "2. Testing health check (GET /health)..."
curl -s "$API_URL/health" | jq '.'
echo ""

# Test 3: Model info
echo "3. Testing model info (GET /model/info)..."
curl -s "$API_URL/model/info" | jq '.'
echo ""

# Test 4: Prediction endpoint
echo "4. Testing prediction endpoint (POST /predict)..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d @examples/api_request_example.json | jq '.'
echo ""

# Test 5: Single prediction
echo "5. Testing single prediction..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | jq '.'
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ API testing completed!"
echo ""
echo "View interactive documentation at:"
echo "  📚 Swagger UI: $API_URL/docs"
echo "  📖 ReDoc:      $API_URL/redoc"
