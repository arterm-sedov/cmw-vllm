#!/bin/bash
# Quick curl test for vLLM

MODEL=${1:-"mistralai/Ministral-3-14B-Instruct-2512"}
QUESTION=${2:-"What is the capital of France?"}

echo "Testing vLLM inference with model: $MODEL"
echo "Question: $QUESTION"
echo ""

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"$QUESTION\"
      }
    ]
  }" | jq .
