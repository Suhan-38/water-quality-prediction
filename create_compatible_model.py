import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

# Print version information for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Note: For Render deployment, use NumPy 1.24.4 and scikit-learn 1.0.2")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"scikit-learn import error: {e}")

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

    # Train a simple model with fewer estimators for faster training
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the model using joblib
    print("Saving model with joblib...")
    joblib.dump(model, 'compatible_model.joblib', compress=3)
    print("Model saved as compatible_model.joblib")

    # Also save with pickle for comparison
    print("Saving model with pickle...")
    with open('compatible_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as compatible_model.pkl")

    # Test loading the model to verify it works
    print("Testing model loading...")
    loaded_model = joblib.load('compatible_model.joblib')
    print("Successfully loaded model with joblib")

    # Verify the loaded model works
    test_accuracy = loaded_model.score(X_test, y_test)
    print(f"Loaded model accuracy: {test_accuracy:.4f}")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
