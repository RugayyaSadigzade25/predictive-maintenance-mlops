import pandas as pd
import mlflow # pyright: ignore[reportMissingImports]
import mlflow.sklearn # pyright: ignore[reportMissingImports]

mlflow.set_tracking_uri(
    "sqlite:///C:/Users/ruqay/OneDrive/predictive-maintenance-mlops/notebooks/mlflow.db"
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Configuration

DATA_PATH = "data/ai4i2020.csv"
EXPERIMENT_NAME = "predictive-maintenance"

# Load the data

print("Loading data...")

data = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {data.shape}")

# Define features and target

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

target = "Machine failure"

X = data[features]
y = data[target]


# Train and test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Preprocessing

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ]
)


# Build model

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)


# MLflow experiment

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():

    print("Training Random Forest...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Results")
    print("--------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Log parameters

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)

    # Log metrics

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


    # Save model to MLflow

    mlflow.sklearn.log_model(
    model,
    name="random_forest_model",
    registered_model_name="predictive-maintenance-model",
    )

    print("\nModel logged to MLflow.")
    print("Training pipeline completed successfully.")