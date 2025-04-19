from flask import Flask, render_template, request, jsonify
import os
import sys

# Print Python version for debugging
print(f"Python version: {sys.version}")

# Try different import strategies
try:
    import joblib
    print("Successfully imported joblib")
except ImportError:
    print("Failed to import joblib, trying pickle")
    import pickle

try:
    import numpy as np
    print(f"Successfully imported NumPy {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {str(e)}")

try:
    import pandas as pd
    print(f"Successfully imported pandas {pd.__version__}")
except ImportError as e:
    print(f"pandas import error: {str(e)}")

app = Flask(__name__)

# Load the trained model
# Try different model files
model = None
model_files = ['compatible_model.joblib', 'random_forest_model.joblib', 'random_forest_model.pkl']

for model_path in model_files:
    if os.path.exists(model_path):
        try:
            print(f"Attempting to load model from {model_path}")
            if model_path.endswith('.joblib'):
                model = joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            print(f"Successfully loaded model from {model_path}")
            break
        except Exception as e:
            print(f"Error loading {model_path}: {str(e)}")

if model is None:
    print("Failed to load any model. Using a fallback model.")
    # Create a simple fallback model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'ph': [float(data['ph'])],
            'Hardness': [float(data['hardness'])],
            'Solids': [float(data['solids'])],
            'Chloramines': [float(data['chloramines'])],
            'Sulfate': [float(data['sulfate'])],
            'Conductivity': [float(data['conductivity'])],
            'Organic_carbon': [float(data['organic_carbon'])],
            'Trihalomethanes': [float(data['trihalomethanes'])],
            'Turbidity': [float(data['turbidity'])]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Get feature importances for this specific prediction
        feature_importances = dict(zip(input_data.columns, model.feature_importances_))
        sorted_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'feature_importances': sorted_importances
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use port 8000 for Docker, fallback to 5000 for local development
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

