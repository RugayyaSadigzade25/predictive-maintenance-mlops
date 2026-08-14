# Predictive Maintenance MLOps

An end-to-end machine learning system for predicting industrial machine failures and monitoring production data for data drift.

This project demonstrates the complete lifecycle of a machine learning application, from data validation and model training to experiment tracking, API deployment, containerization, automated testing, and production monitoring.

## Architecture

![MLOps Architecture](docs/Gemini_Generated_Image_jfcqlbjfcqlbjfcq.png)

## Project Overview

Predictive maintenance is an important application of machine learning in industrial environments, where unexpected equipment failures can lead to downtime, maintenance costs, and production losses.

This project uses industrial sensor data to predict whether a machine is likely to experience a failure. Rather than focusing only on model training, the project implements an end-to-end MLOps workflow.

The system includes:

- Data validation
- Machine failure prediction
- Model training and evaluation
- MLflow experiment tracking
- MLflow Model Registry
- FastAPI model serving
- Docker containerization
- Automated testing with pytest
- GitHub Actions CI
- Production data monitoring
- Statistical data drift detection
- Drift result logging

## Key Features

| Feature | Description |
|---|---|
| Machine Failure Prediction | Predicts whether an industrial machine is likely to fail |
| Data Validation | Validates incoming production data before inference |
| MLflow | Tracks experiments, parameters, and evaluation metrics |
| Model Registry | Maintains trained model versions |
| FastAPI | Provides a REST API for real-time predictions |
| Docker | Packages the application into a reproducible container |
| Monitoring | Collects production observations |
| Drift Detection | Detects changes between reference and production data |
| Automated Testing | Tests API, validation, monitoring, and drift functionality |
| GitHub Actions | Automatically runs the test suite on code changes |

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains industrial machine measurements and machine failure labels.

The model uses the following features:

| Feature | Description |
|---|---|
| Type | Machine type |
| Air temperature [K] | Air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Tool usage time |

The target variable is:

- `Machine failure = 0` — No failure
- `Machine failure = 1` — Machine failure

## Machine Learning Model

A **Random Forest classifier** is used to predict machine failure.

The training pipeline performs preprocessing of categorical and numerical features before training the model. The complete trained pipeline is saved using `joblib` and loaded by the FastAPI application for production inference.

### Model Performance

The final Random Forest model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 97.95% |
| Precision | 71.43% |
| Recall | 66.18% |
| F1-score | 68.70% |

The lower recall for the failure class is particularly important in predictive maintenance because failing to identify an actual machine failure can be more costly than generating a false alarm.

## Experiment Tracking

MLflow is used to track machine learning experiments and model performance.

Training runs record metrics and parameters including:

- Accuracy
- Precision
- Recall
- F1-score
- Model type
- Number of estimators
- Class weighting
- Random state

The project also uses the MLflow Model Registry to manage trained model versions.

This provides a reproducible record of model development and makes it possible to compare different training runs.

### MLflow Experiment Tracking

The Random Forest training run is tracked in MLflow together with its parameters and evaluation metrics.

![MLflow Experiment Tracking](docs/image.png)

## FastAPI

The trained model is served through a **FastAPI REST API**.

The API:

1. Receives machine sensor measurements
2. Validates the incoming data
3. Runs the trained machine learning model
4. Returns the predicted machine failure status
5. Reports the current production data drift status

### API Documentation

FastAPI provides interactive Swagger documentation for testing the available endpoints.

![FastAPI API Documentation](docs/overv.png)

### Prediction Endpoint

The `/predict` endpoint accepts machine sensor measurements and returns the model prediction.

![FastAPI Prediction Request](docs/api.png.png)

![FastAPI Prediction Response](docs/api2.png.png)

## Docker

The application is containerized with Docker to provide a reproducible environment for running the FastAPI inference service.

The Docker image packages the Python dependencies, trained model, application code, and API server into a single deployable unit.

### Running the Application

The container exposes port `8000` and runs the FastAPI application using Uvicorn.

![Docker Container Running](docs/docker.png)

Build the Docker image:

```bash
docker build -t predictive-maintenance-mlops .

Run the container:

docker run -p 8000:8000 predictive-maintenance-mlops

The API documentation is available at:

http://127.0.0.1:8000/docs
Production Monitoring

The system monitors incoming production data and compares it against the reference data distribution.

Production observations are collected in a monitoring window before statistical drift analysis is performed.

The monitoring pipeline is:

Production Data
      ↓
Data Validation
      ↓
FastAPI Prediction
      ↓
Production Monitoring
      ↓
Monitoring Window
      ↓
Drift Detection
      ↓
Monitoring Logs
Data Drift Detection

Drift detection is performed using the Kolmogorov–Smirnov (KS) test.

The system compares reference data with newly collected production data and evaluates whether individual feature distributions have changed significantly.

Detected drift results are recorded by the monitoring logger.

This allows changes in production input data to be identified after deployment.

Testing

The project includes automated tests covering the main components of the ML system.

Tests cover:

FastAPI prediction endpoints
Input data validation
Drift detection
Monitoring logic
Monitoring integration
Monitoring logging
Production monitoring

The current test suite contains 14 tests, all passing successfully.

14 passed

Run the tests with:

python -m pytest
Continuous Integration

GitHub Actions automatically runs the test suite when changes are pushed to the main branch or when a pull request is opened.

The CI workflow is:

Git Push / Pull Request
        ↓
GitHub Actions
        ↓
Set up Python
        ↓
Install Dependencies
        ↓
Run Pytest
        ↓
Pass / Fail

This helps ensure that changes to the project do not break existing functionality.

Project Structure
predictive-maintenance-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│
├── docs/
│   ├── Gemini_Generated_Image_jfcqlbjfcqlbjfcq.png
│   ├── image.png
│   ├── overv.png
│   ├── api.png.png
│   ├── api2.png.png
│   └── docker.png
│
├── model/
│   └── model.joblib
│
├── src/
│   ├── create_production_data.py
│   ├── data_validation.py
│   ├── drift_detection.py
│   ├── monitoring.py
│   ├── monitoring_logger.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validation.py
│   ├── test_monitoring.py
│   ├── test_monitoring_integration.py
│   ├── test_monitoring_logger.py
│   ├── test_monitoring_logging_integration.py
│   └── test_production_monitoring.py
│
├── Dockerfile
├── requirements.txt
└── README.md
Quick Start
Clone the Repository
git clone https://github.com/RugayyaSadigzade25/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
Install Dependencies
pip install -r requirements.txt
Run Tests
python -m pytest
Start the API
uvicorn src.predict:app --reload

Open:

http://127.0.0.1:8000/docs
Run with Docker
docker build -t predictive-maintenance-mlops .
docker run -p 8000:8000 predictive-maintenance-mlops
Technologies
Python
pandas
NumPy
scikit-learn
MLflow
FastAPI
Pydantic
Docker
pytest
GitHub Actions
Future Improvements
Automated model retraining after significant data drift
Model performance monitoring in production
Hyperparameter optimization
Scheduled retraining pipelines
Prometheus/Grafana observability
Cloud deployment
Automated Docker image publishing through CI/CD

