import joblib
import numpy as np
import pandas as pd
import sklearn

# Print version information
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"pandas version: {pd.__version__}")

try:
    # Try to load the model
    model = joblib.load('random_forest_model.joblib')
    print("Successfully loaded the model!")
    
    # Print model info
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances available: {len(model.feature_importances_)}")
    
    # Create a sample input
    sample_data = pd.DataFrame({
        'ph': [7.5],
        'Hardness': [200.0],
        'Solids': [20000.0],
        'Chloramines': [7.0],
        'Sulfate': [400.0],
        'Conductivity': [500.0],
        'Organic_carbon': [12.0],
        'Trihalomethanes': [60.0],
        'Turbidity': [4.0]
    })
    
    # Try to make a prediction
    prediction = model.predict(sample_data)
    print(f"Sample prediction: {prediction}")
    
except Exception as e:
    print(f"Error: {str(e)}")
