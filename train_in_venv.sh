#!/bin/bash
echo "Training model in Python 3.8 virtual environment..."

# Activate the virtual environment
source py38_render_env/bin/activate

# Create a training script
cat > train_render_model.py << 'EOL'
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
EOL

# Run the training script
echo "Running training script..."
python train_render_model.py

# Deactivate the virtual environment
deactivate

echo ""
echo "If the model was created successfully, commit and push the files:"
echo "- render_trained_model.joblib"
echo "- render_trained_model.pkl"
echo ""
echo "Then redeploy your application on Render."

read -p "Press Enter to continue..."
