#!/usr/bin/env bash
# Run ON THE VM after training completes:
#   cd ~/math-tutor && bash start_server.sh
set -euo pipefail

source .venv/bin/activate

echo "=== Checking adapters are present ==="
if [ ! -d "adapters/function_gemma" ]; then
  echo "ERROR: adapters/function_gemma/ not found. Run train.sh first."
  exit 1
fi
if [ ! -d "adapters/gemma12b" ]; then
  echo "ERROR: adapters/gemma12b/ not found. Run train.sh first."
  exit 1
fi
echo "Adapters found."

echo ""
echo "=== Starting Adaptive Math Tutor backend ==="
echo ""
echo "IMPORTANT: In a SECOND terminal, run:"
echo "  ngrok http 8000"
echo "Then give your teammate the wss://<ngrok-id>.ngrok.io/ws URL."
echo ""
echo "Starting uvicorn on port 8000..."

uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
