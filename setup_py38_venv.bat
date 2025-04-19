@echo off
echo Setting up Python 3.8 virtual environment for Render compatibility...

REM Check if Python 3.8 is installed
python -V 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 from https://www.python.org/downloads/release/python-3810/
    pause
    exit /b 1
)

REM Create a virtual environment
echo Creating virtual environment...
python -m venv py38_render_env

REM Activate the virtual environment
call py38_render_env\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install specific versions of packages
echo Installing compatible packages...
pip install wheel
pip install numpy==1.24.4
pip install scikit-learn==1.0.2
pip install pandas==1.3.0
pip install joblib==1.0.1
pip install flask==2.0.1

REM Display installed versions
echo.
echo Installed versions:
pip list | findstr numpy
pip list | findstr scikit-learn
pip list | findstr pandas
pip list | findstr joblib
echo.

echo Virtual environment setup complete.
echo To activate this environment, run: py38_render_env\Scripts\activate
echo.

pause
