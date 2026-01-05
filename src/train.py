import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
local_mlruns_path = os.path.join(project_root, "mlruns")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"Logging to Remote MLflow: {mlflow_uri}")
else:
    # This creates a local 'mlruns' folder in the GitHub Runner
    mlflow.set_tracking_uri("file:./mlruns")
    print("Logging to Local MLflow (Runner storage)")

def train():
    # Load parameters
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    # Load preprocessed data
    df = pd.read_csv("data/processed/churn_cleaned.csv")
    X = df.drop(['churn','phone number'], axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], random_state=config['base']['random_state'], stratify=y
    )

    # Set MLflow experiment
    mlflow.set_experiment("Telecom_Churn_Production")

    with mlflow.start_run():
        # Train Model
        rf = RandomForestClassifier(
            n_estimators=config['train']['n_estimators'],
            max_depth=config['train']['max_depth'],
            class_weight=config['train']['class_weight'],
            random_state=config['base']['random_state']
        )
        rf.fit(X_train, y_train)

        # Apply custom threshold for evaluation
        threshold = config['model_selection']['threshold']
        y_probs = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate Metrics
        metrics = {
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

        # Log to MLflow
        mlflow.log_params(config['train'])
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(rf, "model", registered_model_name="ChurnPredictor")

        print(f"Training Complete. Recall: {metrics['recall']:.2f}, Precision: {metrics['precision']:.2f}")


if __name__ == "__main__":
    train()