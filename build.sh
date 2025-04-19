#!/usr/bin/env bash
set -e

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Print Python version for debugging
python -c "import sys; print(f'Python version: {sys.version}')"

# Install specific versions with binary wheels
echo "Installing NumPy 1.24.4..."
pip install --no-cache-dir numpy==1.24.4

echo "Installing other dependencies..."
pip install --no-cache-dir pandas==1.3.0
pip install --no-cache-dir scikit-learn==1.0.2
pip install --no-cache-dir joblib==1.0.1
pip install --no-cache-dir flask==2.0.1 werkzeug==2.0.1 gunicorn==20.1.0

# Print installed packages for debugging
echo "Installed packages:"
pip list

# Verify NumPy installation
python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); print(f'NumPy path: {np.__path__}'); try: from numpy import _core; print('numpy._core is available') except ImportError: print('numpy._core is NOT available')"

# Verify scikit-learn installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

echo "Build script completed successfully."
