#!/bin/bash
echo "Cleaning up unnecessary files..."

# Remove utility scripts
rm -f check_environment.py
rm -f check_versions.py
rm -f verify_model.py
rm -f convert_model.py

# Remove alternative approach scripts
rm -f create_compatible_model.py
rm -f create_deployment_model.py
rm -f create_render_compatible_model.py
rm -f create_simple_model.py
rm -f train_simple_model.py
rm -f model.py
rm -f train_model_in_docker.py
rm -f train_render_model.py

# Remove Docker and virtual environment setup files
rm -f Dockerfile.train
rm -f docker_build_model.bat
rm -f train_in_docker.bat
rm -f train_in_docker.sh
rm -f train_in_venv.bat
rm -f train_in_venv.sh
rm -f setup_py38_venv.bat
rm -f setup_py38_venv.sh
rm -f setup_render_env.bat
rm -f setup_render_env.sh

# Remove old model files
rm -f random_forest_model.joblib
rm -f random_forest_model.pkl
rm -f compatible_model.joblib
rm -f deployment_model.joblib
rm -f deployment_model.pkl
rm -f simple_model.joblib
rm -f simple_model.pkl
rm -f simple_compatible_model.pkl
rm -f LR_model.sav

# Remove requirements-local.txt (keep only requirements.txt and requirements-render.txt)
rm -f requirements-local.txt

# Remove runtime.txt if not needed
rm -f runtime.txt

echo "Cleanup completed!"
echo
echo "Remaining files:"
ls -la

read -p "Press Enter to continue..."
