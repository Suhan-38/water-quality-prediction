#!/bin/bash
echo "Setting up Python 3.8 virtual environment for Render compatibility..."

# Check if Python 3.8 is installed
if ! command -v python3.8 &> /dev/null; then
    echo "Python 3.8 is not installed or not in PATH."
    echo "Please install Python 3.8 using your package manager:"
    echo "  - Ubuntu/Debian: sudo apt-get install python3.8 python3.8-venv"
    echo "  - macOS: brew install python@3.8"
    read -p "Press Enter to continue..."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3.8 -m venv py38_render_env

# Activate the virtual environment
source py38_render_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install specific versions of packages
echo "Installing compatible packages..."
pip install wheel
pip install numpy==1.24.4
pip install scikit-learn==1.0.2
pip install pandas==1.3.0
pip install joblib==1.0.1
pip install flask==2.0.1

# Display installed versions
echo ""
echo "Installed versions:"
pip list | grep numpy
pip list | grep scikit-learn
pip list | grep pandas
pip list | grep joblib
echo ""

echo "Virtual environment setup complete."
echo "To activate this environment, run: source py38_render_env/bin/activate"
echo ""

read -p "Press Enter to continue..."
