#!/usr/bin/env bash
# Upgrade pip
pip install --upgrade pip

# Install packages with binary wheels first
pip install --only-binary=numpy,scipy,pandas,scikit-learn numpy==1.20.3
pip install --only-binary=numpy,scipy,pandas,scikit-learn pandas==1.3.0
pip install --only-binary=numpy,scipy,pandas,scikit-learn scikit-learn==0.24.2

# Install the rest of the requirements
pip install -r requirements.txt
