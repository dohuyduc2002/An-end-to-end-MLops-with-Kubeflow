#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "🐞 Running unit tests and integration tests in DEBUG mode..."

pytest \
  api/test_prediction_api.py \
  integration/test_pipeline_run.py \
  --log-cli-level=DEBUG \
  --capture=no \
  --tb=long \
  -v

echo "✅ All tests completed!"
