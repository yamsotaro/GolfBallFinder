#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
export HF_HOME="$ROOT/.cache/huggingface"
mkdir -p "$YOLO_CONFIG_DIR" "$HF_HOME"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This bootstrap is intended for macOS because the final iPhone build requires Xcode."
  exit 1
fi

command -v xcodebuild >/dev/null || { echo "Install Xcode first."; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is required by this convenience script: https://brew.sh"; exit 1; }
command -v xcodegen >/dev/null || brew install xcodegen

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m unittest discover -s Tests/Python -v

if [[ ! -d GolfBallFinder/Resources/GolfBall.mlpackage ]]; then
  python scripts/fetch_seed_model.py
  python training/export_coreml.py --weights training/models/seed_golf_ball_yolov8n.pt
fi

xcodegen generate

echo
cat <<'TXT'
Bootstrap complete.
1. Open GolfBallFinder.xcodeproj.
2. Select the GolfBallFinder target > Signing & Capabilities > choose your Team.
3. Connect iPhone 16 Pro, trust the Mac, select it as Run Destination.
4. Run the app.

The included seed model is only for first end-to-end validation. Collect your own rough/grass videos and fine-tune before judging product accuracy.
TXT
