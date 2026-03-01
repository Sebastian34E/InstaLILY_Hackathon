#!/usr/bin/env bash
# Run ON THE VM inside tmux/screen:
#   tmux new -s train
#   cd ~/math-tutor && bash train.sh
set -euo pipefail

source .venv/bin/activate
mkdir -p logs adapters

echo "=== [1/2] Training FunctionGemma 270M (~20 min) ==="
python3 finetune/train_function_gemma.py 2>&1 | tee logs/train_fg.log
echo "FunctionGemma training complete. Adapter saved to adapters/function_gemma/"

echo ""
echo "=== [2/2] Training Gemma 3 12B (~40 min) ==="
python3 finetune/train_gemma12b.py 2>&1 | tee logs/train_12b.log
echo "Gemma 12B training complete. Adapter saved to adapters/gemma12b/"

echo ""
echo "=== Training complete! Run: bash start_server.sh ==="
