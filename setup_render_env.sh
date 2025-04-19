#!/bin/bash
echo "Setting up Render-compatible environment..."

# Create virtual environment
python -m venv render_env
echo "Virtual environment created."

# Activate virtual environment
source render_env/bin/activate
echo "Virtual environment activated."

# Install compatible packages
echo "Installing compatible packages..."
pip install numpy==1.24.4
pip install scikit-learn==1.0.2
pip install pandas==1.3.0
pip install joblib==1.0.1

# Display installed versions
echo ""
echo "Installed versions:"
pip list | grep numpy
pip list | grep scikit-learn
pip list | grep pandas
pip list | grep joblib
echo ""

# Train the model
echo "Training Render-compatible model..."
python create_render_compatible_model.py

echo ""
echo "If the model was created successfully, commit and push the files:"
echo "- render_compatible_model.joblib"
echo "- render_compatible_model.pkl"
echo ""
echo "Then redeploy your application on Render."

read -p "Press Enter to continue..."
