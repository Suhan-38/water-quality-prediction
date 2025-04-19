import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Print NumPy version for reference
print(f"NumPy version: {np.__version__}")

# Load the pickle model
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Successfully loaded the pickle model")
    
    # Save as joblib
    joblib.dump(model, 'random_forest_model.joblib')
    print("Successfully saved the model as joblib format")
    
    # Test loading the joblib model
    loaded_model = joblib.load('random_forest_model.joblib')
    print("Successfully loaded the joblib model")
    
    # Print model info
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances available: {len(model.feature_importances_)}")
    
except Exception as e:
    print(f"Error: {str(e)}")
