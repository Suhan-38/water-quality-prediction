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
# List all files in the current directory for debugging
print("\nListing all files in the current directory:")
for file in os.listdir('.'):
    print(f"- {file} ({os.path.getsize(file)} bytes)")
print()

# Try different model files
model = None
model_files = ['compatible_model.joblib', 'random_forest_model.joblib', 'random_forest_model.pkl']

print("\nAttempting to load models:")
for model_path in model_files:
    print(f"Checking for {model_path}...")
    if os.path.exists(model_path):
        print(f"  File exists: {model_path} ({os.path.getsize(model_path)} bytes)")
        try:
            print(f"  Attempting to load model from {model_path}")
            if model_path.endswith('.joblib'):
                model = joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            print(f"  Successfully loaded model from {model_path}")
            print(f"  Model type: {type(model)}")
            if hasattr(model, 'n_estimators'):
                print(f"  Model n_estimators: {model.n_estimators}")
            if hasattr(model, 'feature_importances_'):
                print(f"  Model has feature_importances_: {len(model.feature_importances_)}")
            break
        except Exception as e:
            print(f"  Error loading {model_path}: {str(e)}")
            print(f"  Error type: {type(e).__name__}")
            # Print traceback for more detailed error information
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
    else:
        print(f"  File does not exist: {model_path}")

if model is None:
    print("Failed to load any model. Creating and training a fallback model.")
    # Create and train a simple fallback model
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        from sklearn.model_selection import train_test_split

        # Try to load the dataset
        try:
            print("Loading dataset for fallback model training...")
            data = pd.read_csv('water_potability.csv')
            # Handle missing values
            data = data.fillna(data.mean())

            # Split features and target
            X = data.drop('Potability', axis=1)
            y = data['Potability']

            # Create and train a simple model
            print("Training fallback model...")
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            print("Fallback model trained successfully.")

            # Save this model for future use
            try:
                import joblib
                joblib.dump(model, 'fallback_model.joblib')
                print("Fallback model saved for future use.")
            except Exception as e:
                print(f"Could not save fallback model: {str(e)}")

        except Exception as e:
            print(f"Could not load dataset: {str(e)}")
            # Create a very simple model with dummy data
            print("Creating a dummy fallback model...")
            import numpy as np
            # Create dummy data with the expected 9 features
            X_dummy = np.random.rand(100, 9)
            y_dummy = np.random.randint(0, 2, 100)
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X_dummy, y_dummy)
            print("Dummy fallback model created.")
    except Exception as e:
        print(f"Failed to create fallback model: {str(e)}")
        # Last resort - create a model that always predicts 0
        class SimpleModel:
            def predict(self, X):
                return np.zeros(len(X))
            def predict_proba(self, X):
                return np.array([[1.0, 0.0] for _ in range(len(X))])
            feature_importances_ = np.array([1/9] * 9)  # Equal importance
        model = SimpleModel()
        print("Created a simple fallback model that always predicts 0.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        print(f"Received prediction request with data: {data}")

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
        print(f"Created input DataFrame with shape: {input_data.shape}")

        # Check if model is fitted
        if not hasattr(model, 'predict'):
            print("Error: Model does not have predict method")
            return jsonify({'error': 'Model not properly initialized'}), 500

        # Make prediction with error handling
        try:
            print("Making prediction...")
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            print(f"Prediction: {prediction}, Probability: {probability}")

            # Get feature importances for this specific prediction
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(input_data.columns, model.feature_importances_))
                sorted_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            else:
                # If model doesn't have feature_importances_, create dummy values
                print("Model doesn't have feature importances, using dummy values")
                sorted_importances = {col: 1.0/len(input_data.columns) for col in input_data.columns}

            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability),
                'feature_importances': sorted_importances
            })
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # If prediction fails, try to train a new model on the fly
            try:
                print("Attempting to create and train a new model...")
                from sklearn.ensemble import RandomForestClassifier
                import numpy as np

                # Create dummy data with the expected 9 features
                X_dummy = np.random.rand(100, 9)
                y_dummy = np.random.randint(0, 2, 100)
                new_model = RandomForestClassifier(n_estimators=5, random_state=42)
                new_model.fit(X_dummy, y_dummy)

                # Try prediction with the new model
                prediction = new_model.predict(input_data)[0]
                probability = new_model.predict_proba(input_data)[0][1]
                feature_importances = dict(zip(input_data.columns, new_model.feature_importances_))
                sorted_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}

                # We'll just use the new model for this request
                # and not try to update the global model

                return jsonify({
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'feature_importances': sorted_importances,
                    'note': 'Used emergency fallback model'
                })
            except Exception as inner_e:
                print(f"Failed to create emergency model: {str(inner_e)}")
                return jsonify({'error': f'Prediction failed: {str(e)}. Emergency model also failed: {str(inner_e)}'}), 500

    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use port 8000 for Docker, fallback to 5000 for local development
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

