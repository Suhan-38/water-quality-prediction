"""
This script checks your environment for compatibility with the water quality prediction application.
It verifies the versions of key dependencies and provides guidance on model compatibility.
"""
import sys
import os
import importlib.util

def check_module_version(module_name):
    """Check if a module is installed and return its version."""
    try:
        if importlib.util.find_spec(module_name) is not None:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                return module.__version__
            else:
                return "Installed (version unknown)"
        else:
            return "Not installed"
    except ImportError:
        return "Not installed"

def check_numpy_core():
    """Check if numpy._core is available (present in NumPy >= 1.26.0)."""
    try:
        from numpy import _core
        return True
    except ImportError:
        return False

def check_model_files():
    """Check which model files exist in the current directory."""
    model_files = [
        'deployment_model.joblib',
        'deployment_model.pkl',
        'compatible_model.joblib',
        'compatible_model.pkl',
        'random_forest_model.joblib',
        'random_forest_model.pkl',
        'fallback_model.joblib'
    ]
    
    existing_files = []
    for model_file in model_files:
        if os.path.exists(model_file):
            existing_files.append(f"{model_file} ({os.path.getsize(model_file)} bytes)")
    
    return existing_files

def main():
    """Main function to check environment and provide guidance."""
    print("\n===== Environment Check for Water Quality Prediction App =====\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check key dependencies
    numpy_version = check_module_version('numpy')
    sklearn_version = check_module_version('sklearn')
    pandas_version = check_module_version('pandas')
    joblib_version = check_module_version('joblib')
    flask_version = check_module_version('flask')
    
    print(f"NumPy version: {numpy_version}")
    print(f"scikit-learn version: {sklearn_version}")
    print(f"pandas version: {pandas_version}")
    print(f"joblib version: {joblib_version}")
    print(f"Flask version: {flask_version}")
    
    # Check for numpy._core
    has_numpy_core = check_numpy_core() if numpy_version != "Not installed" else False
    print(f"numpy._core available: {'Yes' if has_numpy_core else 'No'}")
    
    # Check model files
    print("\nModel files found:")
    model_files = check_model_files()
    if model_files:
        for model_file in model_files:
            print(f"- {model_file}")
    else:
        print("- No model files found")
    
    # Provide guidance
    print("\n===== Compatibility Analysis =====\n")
    
    # Local development
    print("Local Development:")
    if numpy_version != "Not installed" and sklearn_version != "Not installed":
        print("✓ Your environment has the necessary dependencies for local development.")
        
        if has_numpy_core:
            print("✓ You have NumPy >= 1.26.0 with _core module available.")
            print("  - Models saved in this environment may not be compatible with Render deployment.")
            print("  - Run 'python create_deployment_model.py' before deploying to create a compatible model.")
        else:
            print("✓ You have NumPy < 1.26.0 without _core module.")
            print("  - Models saved in this environment should be compatible with Render deployment.")
    else:
        print("✗ Missing key dependencies for local development.")
        print("  - Install required packages using: pip install -r requirements.txt")
    
    # Render deployment
    print("\nRender Deployment:")
    print("- Render uses Python 3.8 with NumPy 1.24.4 and scikit-learn 1.0.2")
    print("- Models saved with NumPy >= 1.26.0 will cause 'No module named numpy._core' error")
    
    if has_numpy_core:
        print("✗ Your current NumPy version is not compatible with Render deployment.")
        print("  - Create a deployment-ready model using: python create_deployment_model.py")
    else:
        print("✓ Your current NumPy version should be compatible with Render deployment.")
    
    # Recommendations
    print("\n===== Recommendations =====\n")
    
    if not model_files:
        print("1. Run 'python model.py' to train and save the initial model")
    
    if has_numpy_core:
        print("2. Run 'python create_deployment_model.py' to create a Render-compatible model")
    
    print("3. For local testing: python app.py")
    print("4. For deployment: Follow the instructions in the README.md file")
    
    print("\n==========================================================\n")

if __name__ == "__main__":
    main()
