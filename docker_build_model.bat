@echo off
echo Building Docker image to create a Render-compatible model...

REM Create a temporary Dockerfile for building the model
echo FROM python:3.8-slim > Dockerfile.model
echo WORKDIR /app >> Dockerfile.model
echo COPY water_potability.csv /app/ >> Dockerfile.model
echo RUN pip install numpy==1.24.4 scikit-learn==1.0.2 pandas==1.3.0 joblib==1.0.1 >> Dockerfile.model
echo COPY create_render_compatible_model.py /app/ >> Dockerfile.model
echo CMD ["python", "create_render_compatible_model.py"] >> Dockerfile.model

REM Build the Docker image
docker build -t render-model-builder -f Dockerfile.model .

REM Run the Docker container to create the model
docker run --name render-model-container -v %cd%:/app/output render-model-builder

REM Copy the model files from the container to the host
docker cp render-model-container:/app/render_compatible_model.joblib .
docker cp render-model-container:/app/render_compatible_model.pkl .

REM Clean up
docker rm render-model-container
del Dockerfile.model

echo.
echo If the model was created successfully, commit and push the files:
echo - render_compatible_model.joblib
echo - render_compatible_model.pkl
echo.
echo Then redeploy your application on Render.

pause
