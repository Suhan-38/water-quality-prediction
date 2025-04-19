"""
This script trains a Random Forest model for water quality prediction
in an environment that matches Render's deployment environment.

To use this script:
1. Create a Python 3.8 virtual environment
2. Install the required packages:
   - numpy==1.24.4
   - scikit-learn==1.0.2
   - pandas==1.3.0
   - joblib==1.0.1
3. Run this script in the virtual environment
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle
import sys

# Print version information
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")
print(f"pandas version: {pd.__version__}")

# Check if numpy._core exists
try:
    from numpy import _core
    print("WARNING: numpy._core exists - this might cause compatibility issues with Render")
except ImportError:
    print("Good: numpy._core does not exist - this is compatible with Render")

# Load the dataset
print("Loading dataset...")
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
joblib.dump(model, 'render_trained_model.joblib')
print("Model saved as render_trained_model.joblib")

# Save with pickle for backup
print("Saving model with pickle...")
with open('render_trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as render_trained_model.pkl")

# Test loading the model to verify it works
print("Testing model loading...")
loaded_model = joblib.load('render_trained_model.joblib')
print("Successfully loaded model with joblib")

# Verify the loaded model works
test_accuracy = loaded_model.score(X_test, y_test)
print(f"Loaded model accuracy: {test_accuracy:.4f}")

print("Model training completed successfully!")
