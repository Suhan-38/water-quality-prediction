import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print(f"NumPy version: {np.__version__}")

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
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model using joblib
    joblib.dump(model, 'compatible_model.joblib')
    print("Model saved as compatible_model.joblib")
    
except Exception as e:
    print(f"Error: {str(e)}")
