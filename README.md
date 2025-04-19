# Water Quality Prediction Application

A beautiful web application that predicts water potability using a Random Forest machine learning algorithm. The application takes various water quality parameters as input and predicts whether the water is safe to drink.

![Water Quality Prediction App](screenshots/app_screenshot.png)

## Features

- **Beautiful UI**: Clean, responsive interface with modern design
- **Machine Learning Backend**: Random Forest algorithm for accurate predictions
- **Interactive Form**: Input fields for all water quality parameters with helpful tooltips
- **Real-time Prediction**: Instant prediction of water potability
- **Visual Results**: Graphical display of prediction results and feature importance
- **Sample Data**: Option to use sample data for quick testing

## Table of Contents

- [Installation](#installation)
  - [For Users with Python Already Installed](#for-users-with-python-already-installed)
  - [For Users without Python](#for-users-without-python)
- [Deployment](#deployment)
  - [Deploying to Render](#deploying-to-render)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Model Information](#model-information)
- [Technologies Used](#technologies-used)
- [License](#license)

## Installation

### For Users with Python Already Installed

If you already have Python installed on your system, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/water-quality-prediction.git
   cd water-quality-prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser and go to**
   ```
   http://localhost:5000
   ```

### For Users without Python

If you don't have Python installed on your system, follow these steps:

1. **Download and Install Python**
   - Go to [Python's official website](https://www.python.org/downloads/)
   - Download the latest version of Python for your operating system
   - Run the installer
   - **Important**: Check the box that says "Add Python to PATH" during installation
   - Complete the installation

2. **Download the project**
   - Download the ZIP file of this repository
   - Extract the ZIP file to a folder of your choice

3. **Open Command Prompt or Terminal**
   - On Windows: Press Win+R, type "cmd" and press Enter
   - On macOS: Open Terminal from Applications > Utilities
   - On Linux: Open Terminal

4. **Navigate to the project directory**
   ```bash
   cd path/to/extracted/folder
   ```

5. **Create a virtual environment**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

6. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

7. **Train the model**
   ```bash
   python model.py
   ```

8. **Run the application**
   ```bash
   python app.py
   ```

9. **Open your browser and go to**
   ```
   http://localhost:5000
   ```

## Deployment

### Deploying to Render

This application can be easily deployed to Render. Follow these steps:

1. **Create a Render account**
   - Go to [Render](https://render.com/) and sign up for an account

2. **Create a new Web Service**
   - Click on "New" and select "Web Service"
   - Connect your GitHub repository or use the public repository URL

3. **Configure the Web Service**
   - Name: Choose a name for your application (e.g., "water-quality-prediction")
   - Environment: Select "Docker"
   - Branch: Choose the branch to deploy (usually "main" or "master")
   - Root Directory: Leave empty if your Dockerfile is in the root directory
   - Plan: Select the free plan for testing

4. **Advanced Settings**
   - No additional environment variables are required

5. **Click "Create Web Service"**
   - Render will automatically detect the Dockerfile and build your application

6. **Important Version Compatibility Notes**
   - For Render deployment, the application uses NumPy 1.24.4 and scikit-learn 1.0.2 (specified in requirements-render.txt)
   - For local development, you can use newer versions of these libraries
   - The model serialization/deserialization may have compatibility issues between different NumPy versions
   - If you encounter the error `ModuleNotFoundError: No module named 'numpy._core'`, it means your model was saved with a newer NumPy version than what's available in the deployment environment
   - Solution: Run `python create_deployment_model.py` locally before deploying to create a compatible model

7. **Access Your Deployed Application**
   - Once the deployment is complete, Render will provide a URL to access your application
   - The URL will look like `https://your-app-name.onrender.com`

## Usage

1. Once the application is running, you'll see a form with input fields for various water quality parameters.

2. Enter values for all the parameters:
   - **pH Level**: Measure of how acidic/basic water is (0-14)
   - **Hardness**: Capacity of water to precipitate soap (mg/L)
   - **Solids (TDS)**: Total dissolved solids (ppm)
   - **Chloramines**: Amount of Chloramines (ppm)
   - **Sulfate**: Amount of Sulfates dissolved (mg/L)
   - **Conductivity**: Electrical conductivity of water (μS/cm)
   - **Organic Carbon**: Amount of organic carbon (ppm)
   - **Trihalomethanes**: Amount of Trihalomethanes (μg/L)
   - **Turbidity**: Measure of light emitting property of water (NTU)

3. Click "Predict Potability" to get the prediction result.

4. Alternatively, you can click "Use Sample Data" to fill the form with sample values.

5. View the prediction result and feature importance chart to understand which parameters have the most significant impact on water potability.

## Dataset Information

The dataset used for training the model contains water quality metrics for 3,276 different water bodies. Each water sample has the following attributes:

1. **pH**: pH of the water (0 to 14)
2. **Hardness**: Capacity of water to precipitate soap in mg/L
3. **Solids**: Total dissolved solids in ppm
4. **Chloramines**: Amount of Chloramines in ppm
5. **Sulfate**: Amount of Sulfates dissolved in mg/L
6. **Conductivity**: Electrical conductivity of water in μS/cm
7. **Organic_carbon**: Amount of organic carbon in ppm
8. **Trihalomethanes**: Amount of Trihalomethanes in μg/L
9. **Turbidity**: Measure of light emitting property of water in NTU
10. **Potability**: Indicates if water is safe for human consumption (0 = Not Potable, 1 = Potable)

## Model Information

The application uses a Random Forest Classifier from scikit-learn with the following characteristics:

- **Algorithm**: Random Forest
- **Number of Trees**: 100
- **Feature Importance**: The model provides insights into which water parameters have the most significant impact on potability
- **Accuracy**: The model achieves high accuracy on the test dataset

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python Flask
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Custom CSS and JavaScript

## License

This project is licensed under the MIT License - see the LICENSE file for details.
