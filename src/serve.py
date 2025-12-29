# src/serve.py
import pickle
import pandas as pd
import mlflow
import uvicorn
import xgboost as xgb
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "nyc-taxi-fare-predictor"
MODEL_STAGE = "Staging"  # Load from 'Staging' stage. Change to 'Production' for final deployment.

# --- Setup MLflow ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


# --- Helper to load the correct model ---
def load_production_model():
    """
    Loads the latest model AND its preprocessor from the MLflow Model Registry.
    Returns: (xgb_model, preprocessor, model_version)
    """
    try:
        # Get the latest model version in the specified stage
        model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not model_version:
            raise ValueError(f"No model found with name '{MODEL_NAME}' in stage '{MODEL_STAGE}'.")

        latest_version = model_version[0]
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

        print(f"✅ Loading model: {MODEL_NAME}, Version: {latest_version.version}, Stage: {MODEL_STAGE}")

        # 1. Load the XGBoost model directly (not as pyfunc)
        xgb_model = mlflow.xgboost.load_model(model_uri)

        # 2. Download the preprocessor artifact from the SAME run
        #    The preprocessor was logged at artifact_path="preprocessor/preprocessor.b"
        run_id = latest_version.run_id
        preprocessor_uri = f"runs:/{run_id}/preprocessor/preprocessor.b"
        local_path = mlflow.artifacts.download_artifacts(preprocessor_uri)

        with open(local_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return xgb_model, preprocessor, latest_version.version

    except Exception as e:
        print(f"❌ Error loading model from registry: {e}")
        # Fallback to local files if needed (for development)
        print("⚠️  Attempting to load local model and preprocessor...")
        xgb_model = xgb.Booster()
        xgb_model.load_model("models/model.xgb")
        with open("models/preprocessor.b", "rb") as f:
            preprocessor = pickle.load(f)
        return xgb_model, preprocessor, "local_fallback"


# --- Load model and preprocessor at startup ---
model, preprocessor, model_version = load_production_model()

# --- FastAPI App (remainder stays the same, but update the predict function) ---
app = FastAPI(title="NYC Taxi Duration Predictor", version=model_version)


class PredictionRequest(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_version: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 1. Prepare the input EXACTLY as in training
        input_dict = request.dict()
        input_dict["PU_DO"] = f"{input_dict['PULocationID']}_{input_dict['DOLocationID']}"

        # 2. Transform using the SAME preprocessor
        # Note: preprocessor expects a list of dicts
        input_features = [{
            "PU_DO": input_dict["PU_DO"],
            "trip_distance": input_dict["trip_distance"]
        }]
        X = preprocessor.transform(input_features)  # This creates the numeric matrix

        # 3. Create DMatrix and predict
        dmatrix = xgb.DMatrix(X)
        prediction = model.predict(dmatrix)

        return PredictionResponse(
            predicted_duration_minutes=float(prediction[0]),
            model_version=f"v{model_version}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "NYC Taxi Trip Duration Prediction API", "model_stage": MODEL_STAGE}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    Expects JSON with PULocationID, DOLocationID, and trip_distance.
    """
    try:
        # 1. Prepare the input data (replicate training feature engineering)
        input_dict = request.dict()
        input_dict["PU_DO"] = f"{input_dict['PULocationID']}_{input_dict['DOLocationID']}"

        # Create a DataFrame with the expected feature names
        # The model expects the same features it was trained on.
        input_df = pd.DataFrame([{
            "PU_DO": input_dict["PU_DO"],
            "trip_distance": input_dict["trip_distance"]
        }])

        # 2. Make prediction
        # The pyfunc model's .predict() expects a DataFrame
        prediction = model.predict(input_df)

        # 3. Return response
        return PredictionResponse(
            predicted_duration_minutes=float(prediction[0]),
            model_version=f"v{model_version}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# --- For local testing ---
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)