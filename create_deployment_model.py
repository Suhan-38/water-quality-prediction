"""
This script creates a model specifically for deployment to Render.
It uses a simpler model structure to avoid compatibility issues.
"""
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# Print version information for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"scikit-learn import error: {e}")

# Load the dataset
try:
    # Check if dataset exists
    if not os.path.exists('water_potability.csv'):
        print("Dataset not found. Creating dummy data...")
        # Create dummy data with the expected 9 features
        np.random.seed(42)
        X_dummy = np.random.rand(3000, 9)
        y_dummy = np.random.randint(0, 2, 3000)

        # Create a DataFrame with the expected column names
        columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                  'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        X = pd.DataFrame(X_dummy, columns=columns)
        y = pd.Series(y_dummy, name='Potability')

        print("Dummy data created successfully.")
    else:
        # Load the real dataset
        data = pd.read_csv('water_potability.csv')
        print(f"Dataset loaded successfully with shape: {data.shape}")

        # Handle missing values
        data = data.fillna(data.mean())

        # Split features and target
        X = data.drop('Potability', axis=1)
        y = data['Potability']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model with fewer estimators for faster training and better compatibility
    # Using parameters compatible with scikit-learn 1.0.2
    model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
    print("Training deployment model...")
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the model using joblib with lower compression
    print("Saving deployment model with joblib...")
    joblib.dump(model, 'deployment_model.joblib', compress=1)
    print("Model saved as deployment_model.joblib")

    # Also save with pickle for backup
    print("Saving deployment model with pickle...")
    with open('deployment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as deployment_model.pkl")

    # Test loading the model to verify it works
    print("Testing model loading...")
    loaded_model = joblib.load('deployment_model.joblib')
    print("Successfully loaded model with joblib")

    # Verify the loaded model works
    test_accuracy = loaded_model.score(X_test, y_test)
    print(f"Loaded model accuracy: {test_accuracy:.4f}")

    print("\nDeployment model creation completed successfully!")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
