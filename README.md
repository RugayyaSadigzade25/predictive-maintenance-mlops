# Predictive Maintenance MLOps

An end-to-end machine learning and MLOps project for predicting machine failures from industrial sensor data.

The project demonstrates the complete lifecycle of a machine learning model, from data preparation and model training to experiment tracking, model registration, API deployment, containerization, automated testing, and continuous integration.

![CI](https://github.com/RugayyaSadigzade25/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)

---

## Tech Stack

- Python 3.11
- pandas
- NumPy
- scikit-learn
- MLflow
- FastAPI
- Pydantic
- Docker
- pytest
- Git
- GitHub Actions

---

## Project Overview

Predictive maintenance uses machine data to identify potential equipment failures before they occur.

In this project, a machine learning model is trained using industrial sensor measurements such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

The trained model predicts whether a machine failure is likely to occur.

The project then takes the trained model beyond the notebook environment and integrates it into an MLOps pipeline:

1. Data preparation and model training
2. Experiment tracking with MLflow
3. Model registration with MLflow Model Registry
4. REST API deployment using FastAPI
5. Containerization using Docker
6. Automated testing with pytest
7. Continuous integration using GitHub Actions

---

## Architecture

```text
                    ┌─────────────────────┐
                    │  Industrial Data    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preparation &  │
                    │ Model Training      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       MLflow        │
                    │ Tracking + Registry │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI API      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Docker Container  │
                    └─────────────────────┘


        ┌──────────────────────────────────────────┐
        │        Development / Quality Layer       │
        │                                          │
        │     pytest  ───────►  GitHub Actions     │
        │                           CI             │
        └──────────────────────────────────────────┘