#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="diabetic-readmission"
PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "Setting up $PROJECT_NAME..."

# Check Python version
$PYTHON -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'" || exit 1

# Create venv
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- please edit with your actual values"
fi

# Create data/model directories
mkdir -p data models

echo ""
echo "Setup complete!"
echo "Activate with: source $VENV_DIR/bin/activate"
