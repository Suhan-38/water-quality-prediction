"""
This script trains a simplified Random Forest model that will be compatible with any environment.
It uses a smaller number of trees and simpler parameters to ensure compatibility.
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

# Train a simplified Random Forest model with fewer trees and simpler parameters
print("Training simplified Random Forest model...")
model = RandomForestClassifier(
    n_estimators=20,  # Fewer trees for simplicity
    max_depth=10,     # Limit depth for simpler model
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model using joblib with protocol=4 (compatible with Python 3.8)
print("Saving model with joblib...")
joblib.dump(model, 'simple_compatible_model.joblib', compress=1, protocol=4)
print("Model saved as simple_compatible_model.joblib")

# Save with pickle for backup with protocol=4 (compatible with Python 3.8)
print("Saving model with pickle...")
with open('simple_compatible_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
print("Model saved as simple_compatible_model.pkl")

# Test loading the model to verify it works
print("Testing model loading...")
loaded_model = joblib.load('simple_compatible_model.joblib')
print("Successfully loaded model with joblib")

# Verify the loaded model works
test_accuracy = loaded_model.score(X_test, y_test)
print(f"Loaded model accuracy: {test_accuracy:.4f}")

print("\nModel training completed successfully!")
print("The model files are saved and ready for deployment.")
