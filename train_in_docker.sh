#!/bin/bash
echo "Building Docker image for training the model in a Render-compatible environment..."

# Build the Docker image
docker build -t render-model-trainer -f Dockerfile.train .

# Create a directory for the output
mkdir -p model_output

# Run the Docker container with a volume mount to save the model
docker run --name render-model-container -v $(pwd)/model_output:/app render-model-trainer

# Copy the model files from the container to the current directory
cp model_output/render_trained_model.joblib .
cp model_output/render_trained_model.pkl .

# Clean up
docker rm render-model-container
rm -rf model_output

echo ""
echo "If the model was created successfully, commit and push the files:"
echo "- render_trained_model.joblib"
echo "- render_trained_model.pkl"
echo ""
echo "Then redeploy your application on Render."

read -p "Press Enter to continue..."
