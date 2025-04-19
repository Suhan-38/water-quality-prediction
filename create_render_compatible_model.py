"""
This script trains a model using the same versions of NumPy and scikit-learn as Render.
Run this in a Python 3.8 environment with NumPy 1.24.4 and scikit-learn 1.0.2.

Setup instructions:
1. Create a virtual environment: python -m venv render_env
2. Activate the environment:
   - Windows: render_env\\Scripts\\activate
   - Mac/Linux: source render_env/bin/activate
3. Install dependencies: pip install numpy==1.24.4 scikit-learn==1.0.2 pandas joblib
4. Run this script: python create_render_compatible_model.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle
import sys
import os

# Print version information
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn not installed")

# Check if numpy._core exists (it shouldn't in NumPy 1.24.4)
try:
    from numpy import _core
    print("WARNING: numpy._core exists - this might cause compatibility issues with Render")
except ImportError:
    print("Good: numpy._core does not exist - this is compatible with Render")

# Load the dataset
try:
    data = pd.read_csv('water_potability.csv')
    print(f"Dataset loaded successfully with shape: {data.shape}")
    
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Split features and target
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model using joblib
    print("Saving model with joblib...")
    joblib.dump(model, 'render_compatible_model.joblib')
    print("Model saved as render_compatible_model.joblib")
    
    # Save with pickle for backup
    print("Saving model with pickle...")
    with open('render_compatible_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as render_compatible_model.pkl")
    
    # Test loading the model to verify it works
    print("Testing model loading...")
    loaded_model = joblib.load('render_compatible_model.joblib')
    print("Successfully loaded model with joblib")
    
    # Verify the loaded model works
    test_accuracy = loaded_model.score(X_test, y_test)
    print(f"Loaded model accuracy: {test_accuracy:.4f}")
    
    print("\nModel creation completed successfully!")
    print("\nIMPORTANT: Push these model files to your repository:")
    print("- render_compatible_model.joblib")
    print("- render_compatible_model.pkl")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
