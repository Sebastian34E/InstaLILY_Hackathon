#!/usr/bin/env bash
# Run from your local machine: bash backend/deploy.sh
# Deploys backend code to RTX 6000 and sets up the environment.
set -euo pipefail

VM="hackathon@34.133.228.240"
LOCAL_BACKEND="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Checking VM state ==="
ssh "$VM" bash <<'ENDSSH'
  echo '--- HuggingFace cache ---'
  ls ~/.cache/huggingface/hub 2>/dev/null || echo '(empty or missing)'
  echo '--- math-tutor dir ---'
  ls ~/math-tutor 2>/dev/null || echo '(does not exist yet)'
ENDSSH

echo ""
echo "=== Step 2: Copying backend code to VM ==="
ssh "$VM" 'mkdir -p ~/math-tutor/logs ~/math-tutor/adapters'
(cd "$LOCAL_BACKEND" && scp -r . "$VM:~/math-tutor/")
echo "Code copied."

echo ""
echo "=== Step 3: Setting up Python environment ==="
ssh "$VM" bash <<'ENDSSH'
  sudo apt-get install -y python3-venv python3-pip 2>/dev/null || echo "(apt unavailable, skipping)"
  cd ~/math-tutor
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r ~/math-tutor/requirements.txt
  echo 'Dependencies installed.'
ENDSSH

echo ""
echo "=== Step 4: Generating training data ==="
ssh "$VM" bash <<'ENDSSH'
  cd ~/math-tutor
  source .venv/bin/activate
  python3 finetune/generate_data.py
  echo 'Training data generated.'
ENDSSH

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps — SSH into VM and run:"
echo "  ssh <VM>"
echo "  cd ~/math-tutor"
echo "  bash train.sh          # ~60 min, run in tmux or screen"
echo "  bash start_server.sh   # after training completes"
echo ""
echo "To watch training progress:"
echo "  tail -f ~/math-tutor/logs/train_fg.log"
echo "  tail -f ~/math-tutor/logs/train_12b.log"
