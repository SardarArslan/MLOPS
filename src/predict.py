import pandas as pd
import mlflow.sklearn
import os
import yaml


def run_batch_prediction():
    # 1. Load Config
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    # 2. Set MLflow URI (Use local for now, env var for cloud)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # 3. Load the model from the Model Registry (Staging version)
    model_name = "ChurnPredictor"
    stage = "Staging"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")

    # 4. Load "New" customers for prediction
    # (In real life, this is the daily snapshot from your DB)
    df = pd.read_csv("data/processed/churn_cleaned.csv").drop(columns=['churn'])

    # 5. Predict
    # We use predict_proba because Marketing teams like to prioritize the "most likely" to leave
    probs = model.predict_proba(df)[:, 1]

    # 6. Store Predictions
    results = pd.DataFrame({
        'customer_id': df.index,  # Replace with actual ID column if available
        'churn_probability': probs
    })

    os.makedirs("predictions", exist_ok=True)
    results.to_csv("predictions/daily_scores.csv", index=False)
    print(f"Predictions saved for {len(results)} customers.")


if __name__ == "__main__":
    run_batch_prediction()