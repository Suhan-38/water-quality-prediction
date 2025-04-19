@echo off
echo Building Docker image for training the model in a Render-compatible environment...

REM Build the Docker image
docker build -t render-model-trainer -f Dockerfile.train .

REM Create a directory for the output
mkdir model_output

REM Run the Docker container with a volume mount to save the model
docker run --name render-model-container -v %cd%/model_output:/app render-model-trainer

REM Copy the model files from the container to the current directory
copy model_output\render_trained_model.joblib .
copy model_output\render_trained_model.pkl .

REM Clean up
docker rm render-model-container
rmdir /s /q model_output

echo.
echo If the model was created successfully, commit and push the files:
echo - render_trained_model.joblib
echo - render_trained_model.pkl
echo.
echo Then redeploy your application on Render.

pause
