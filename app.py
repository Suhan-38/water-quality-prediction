from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = 'random_forest_model.pkl'

# Check if model exists, if not train it
if not os.path.exists(model_path):
    import model  # This will train and save the model

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

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
    app.run(debug=True)
