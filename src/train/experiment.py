import pickle
import json
import yaml
import os
import mlflow
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import numpy as np

# Load the processed data
print("Loading processed data...")
with open("data/processed/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("data/processed/X_val.pkl", "rb") as f:
    X_val = pickle.load(f)

with open("data/processed/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("data/processed/y_val.pkl", "rb") as f:
    y_val = pickle.load(f)

with open("data/processed/dv.pkl", "rb") as f:
    dv = pickle.load(f)

print(f"Data loaded: X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# Create DMatrix
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("taxi_trip")


def hyp_tuning():
    """Hyperparameter tuning with MLflow tracking"""

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42
    }

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            return {"loss": rmse, "status": STATUS_OK}

    print("Starting hyperparameter tuning...")
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Reduced for testing
        trials=Trials()
    )

    print(f"Best parameters found: {best_result}")
    return best_result


def train_final_model(best_params=None):
    """Train final model with best parameters"""

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    if best_params is None:
        # Use your pre-tuned parameters
        params = {
            "learning_rate": 0.1291168559115615,
            "max_depth": 11,
            "min_child_weight": 2.1202867152859155,
            "objective": "reg:squarederror",
            "reg_alpha": 0.0617417825060163,
            "reg_lambda": 0.10052786055614543,
            "seed": 42
        }
    else:
        params = best_params
        params["objective"] = "reg:squarederror"
        params["seed"] = 42

    with mlflow.start_run(run_name="final_model") as run:
        print(f"Training final model with params: {params}")
        mlflow.set_tag("model", "xgboost_final")
        mlflow.log_params(params)

        # Train model
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # Predict and evaluate
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        print(f"Final model RMSE: {rmse}")

        # Save model artifacts
        model_path = "models/model.xgb"
        booster.save_model(model_path)

        # Save preprocessor
        preprocessor_path = "models/preprocessor.b"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(dv, f)

        # Log artifacts to MLflow
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="model")

        # Register model
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "nyc-taxi-fare-predictor"
        )

        # Save metrics for DVC
        metrics = {
            "rmse": float(rmse),
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model"
        }

        with open("metrics/rmse.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return rmse, run.info.run_id


if __name__ == "__main__":
    # Option 1: Run hyperparameter tuning first
    # best_params = hyp_tuning()
    # rmse, run_id = train_final_model(best_params)


    # Load parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
    # Option 2: Just train with known parameters (faster for testing)
    rmse, run_id = train_final_model(params)

    print(f"\n✅ Training complete!")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Run ID: {run_id}")
    print(f"   Model saved to: models/model.xgb")
    print(f"   Metrics saved to: metrics/rmse.json")