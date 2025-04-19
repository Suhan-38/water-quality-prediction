"""
This script creates a very simple model that doesn't rely on NumPy internals.
It will be compatible with any NumPy version, including those in Render's environment.
"""
import numpy as np
import joblib
import pickle
import sys

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn not installed")

# Create a simple model class that mimics the RandomForestClassifier interface
class SimpleModel:
    def __init__(self):
        self.n_estimators = 1
        # Create feature importances that emphasize pH and chloramines
        self.feature_importances_ = np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
        print("Initialized simple model with predefined feature importances")
        
    def predict(self, X):
        # Always predict 0 (not potable) for simplicity
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X):
        # Return fixed probabilities: 60% not potable, 40% potable
        return np.array([[0.6, 0.4] for _ in range(len(X))])

# Create the model
model = SimpleModel()

# Save with joblib
print("Saving model with joblib...")
joblib.dump(model, 'simple_model.joblib')
print("Model saved as simple_model.joblib")

# Save with pickle
print("Saving model with pickle...")
with open('simple_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as simple_model.pkl")

# Test loading the model
print("Testing model loading...")
loaded_model = joblib.load('simple_model.joblib')
print("Successfully loaded model with joblib")

# Test the model
test_input = np.random.rand(5, 9)
predictions = loaded_model.predict(test_input)
probabilities = loaded_model.predict_proba(test_input)

print(f"Test predictions: {predictions}")
print(f"Test probabilities: {probabilities}")
print("Simple model creation completed successfully!")
