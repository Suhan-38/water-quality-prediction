#!/usr/bin/env bash
set -e

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install specific versions with binary wheels
pip install --no-cache-dir --only-binary=:all: numpy==1.20.3
pip install --no-cache-dir --only-binary=:all: pandas==1.3.0
pip install --no-cache-dir --only-binary=:all: scikit-learn==0.24.2
pip install --no-cache-dir flask==2.0.1 werkzeug==2.0.1 gunicorn==20.1.0 pickle-mixin==1.0.2

# Print installed packages for debugging
pip list
