@echo off
echo Training model in Python 3.8 virtual environment...

REM Activate the virtual environment
call py38_render_env\Scripts\activate

REM Create a training script
echo import numpy as np > train_render_model.py
echo import pandas as pd >> train_render_model.py
echo from sklearn.ensemble import RandomForestClassifier >> train_render_model.py
echo from sklearn.model_selection import train_test_split >> train_render_model.py
echo import joblib >> train_render_model.py
echo import pickle >> train_render_model.py
echo import sys >> train_render_model.py
echo. >> train_render_model.py
echo # Print version information >> train_render_model.py
echo print(f"Python version: {sys.version}") >> train_render_model.py
echo print(f"NumPy version: {np.__version__}") >> train_render_model.py
echo print(f"scikit-learn version: {sklearn.__version__}") >> train_render_model.py
echo print(f"pandas version: {pd.__version__}") >> train_render_model.py
echo. >> train_render_model.py
echo # Check if numpy._core exists >> train_render_model.py
echo try: >> train_render_model.py
echo     from numpy import _core >> train_render_model.py
echo     print("WARNING: numpy._core exists - this might cause compatibility issues with Render") >> train_render_model.py
echo except ImportError: >> train_render_model.py
echo     print("Good: numpy._core does not exist - this is compatible with Render") >> train_render_model.py
echo. >> train_render_model.py
echo # Load the dataset >> train_render_model.py
echo print("Loading dataset...") >> train_render_model.py
echo data = pd.read_csv('water_potability.csv') >> train_render_model.py
echo print(f"Dataset loaded successfully with shape: {data.shape}") >> train_render_model.py
echo. >> train_render_model.py
echo # Handle missing values >> train_render_model.py
echo data = data.fillna(data.mean()) >> train_render_model.py
echo. >> train_render_model.py
echo # Split features and target >> train_render_model.py
echo X = data.drop('Potability', axis=1) >> train_render_model.py
echo y = data['Potability'] >> train_render_model.py
echo. >> train_render_model.py
echo # Split data >> train_render_model.py
echo X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) >> train_render_model.py
echo. >> train_render_model.py
echo # Train a Random Forest model >> train_render_model.py
echo print("Training Random Forest model...") >> train_render_model.py
echo model = RandomForestClassifier(n_estimators=100, random_state=42) >> train_render_model.py
echo model.fit(X_train, y_train) >> train_render_model.py
echo. >> train_render_model.py
echo # Evaluate >> train_render_model.py
echo accuracy = model.score(X_test, y_test) >> train_render_model.py
echo print(f"Model accuracy: {accuracy:.4f}") >> train_render_model.py
echo. >> train_render_model.py
echo # Save the model using joblib >> train_render_model.py
echo print("Saving model with joblib...") >> train_render_model.py
echo joblib.dump(model, 'render_trained_model.joblib') >> train_render_model.py
echo print("Model saved as render_trained_model.joblib") >> train_render_model.py
echo. >> train_render_model.py
echo # Save with pickle for backup >> train_render_model.py
echo print("Saving model with pickle...") >> train_render_model.py
echo with open('render_trained_model.pkl', 'wb') as f: >> train_render_model.py
echo     pickle.dump(model, f) >> train_render_model.py
echo print("Model saved as render_trained_model.pkl") >> train_render_model.py
echo. >> train_render_model.py
echo # Test loading the model to verify it works >> train_render_model.py
echo print("Testing model loading...") >> train_render_model.py
echo loaded_model = joblib.load('render_trained_model.joblib') >> train_render_model.py
echo print("Successfully loaded model with joblib") >> train_render_model.py
echo. >> train_render_model.py
echo # Verify the loaded model works >> train_render_model.py
echo test_accuracy = loaded_model.score(X_test, y_test) >> train_render_model.py
echo print(f"Loaded model accuracy: {test_accuracy:.4f}") >> train_render_model.py
echo. >> train_render_model.py
echo print("Model training completed successfully!") >> train_render_model.py

REM Run the training script
echo Running training script...
python train_render_model.py

REM Deactivate the virtual environment
deactivate

echo.
echo If the model was created successfully, commit and push the files:
echo - render_trained_model.joblib
echo - render_trained_model.pkl
echo.
echo Then redeploy your application on Render.

pause
