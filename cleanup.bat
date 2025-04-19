@echo off
echo Cleaning up unnecessary files...

REM Remove utility scripts
del check_environment.py
del check_versions.py
del verify_model.py
del convert_model.py

REM Remove alternative approach scripts
del create_compatible_model.py
del create_deployment_model.py
del create_render_compatible_model.py
del create_simple_model.py
del train_simple_model.py
del model.py
del train_model_in_docker.py
del train_render_model.py

REM Remove Docker and virtual environment setup files
del Dockerfile.train
del docker_build_model.bat
del train_in_docker.bat
del train_in_docker.sh
del train_in_venv.bat
del train_in_venv.sh
del setup_py38_venv.bat
del setup_py38_venv.sh
del setup_render_env.bat
del setup_render_env.sh

REM Remove old model files
del random_forest_model.joblib
del random_forest_model.pkl
del compatible_model.joblib
del deployment_model.joblib
del deployment_model.pkl
del simple_model.joblib
del simple_model.pkl
del simple_compatible_model.pkl
del LR_model.sav

REM Remove requirements-local.txt (keep only requirements.txt and requirements-render.txt)
del requirements-local.txt

REM Remove runtime.txt if not needed
del runtime.txt

echo Cleanup completed!
echo.
echo Remaining files:
dir /b

pause
