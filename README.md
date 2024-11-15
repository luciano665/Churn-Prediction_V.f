
# Churn Prediction Project
## State of deployment
The lunch time using the AWS EC2 instance was from: 2024/11/07 16:57 GMT-5 to 2024/11/15 GTM-5 8:48. Due to another current project being develop the app is not running 
## Overview
The current project is deployed using AWS and docker.
This project provides a machine-learning model for predicting customer churn. It includes the necessary scripts and configurations to deploy the model as a web application, suitable for Docker-based or local deployment. The project leverages Python with dependencies managed through Poetry.

## Directory Structure

- **`app.py`**: Main application file to run the churn prediction model or API.
- **`Dockerfile`**: Contains instructions for setting up a Docker container to run the application.
- **`requirements.txt`**: Lists the Python dependencies.
- **`utils.py`**: Contains utility functions supporting the main application.

## Prerequisites

- **Python**: Ensure Python is installed (recommended: Python 3.8 or above).
- **Poetry**: Used for dependency management. Install with `pip install poetry`.
- **Docker**: Optional, for containerized deployment.

## Installation

1. Clone the repository.
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   ```bash
   pip install -r requirements.txt
   ```
## manual installation with requirements.txt:
```bash
pip install -r requirements.txt
```
## Running the Application
To start the application:

```bash
python app.py
```
## For Docker users:

Build the Docker image:
```bash
docker build -t churn-prediction .
```
## Run the Docker container:
```bash
docker run -p 8000:8000 churn-prediction
```
## Usage
Once the application is running, access it locally or through the specified port if using Docker.

## Additional Files
 - .gitignore: Lists files to exclude from version control.
 - .replit: Contains configuration for running the app on Replit.
 - poetry.lock, pyproject.toml: Configuration files for Poetry.

