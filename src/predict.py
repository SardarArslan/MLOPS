import pandas as pd
import mlflow
import yaml
import os
import json
from datetime import datetime
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import ColumnDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import logging
import boto3
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_prediction(date=None):
    """Main function to run daily predictions and monitoring"""

    # 1. SETUP AND CONFIGURATION
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')

    logger.info(f"🚀 Starting prediction pipeline for {date}")

    # Load configuration
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    threshold = config['model_selection'].get('threshold', 0.5)

    # Create necessary directories
    os.makedirs("predictions/reports", exist_ok=True)
    os.makedirs("predictions/data", exist_ok=True)

    # 2. LOAD MODEL FROM MLFLOW
    model_name = "ChurnPredictor"
    model_stage = "Staging"  # Use Staging for daily predictions

    model = None
    for stage in ["Staging", "Production"]:
        try:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Attempting to load model from {stage} stage...")
            model = mlflow.pyfunc.load_model(model_uri)
            model_stage = stage
            logger.info(f"✅ Successfully loaded {model_name} from {stage} stage")
            break
        except Exception as e:
            logger.warning(f"Could not load model from {stage}: {e}")
            continue

    if model is None:
        raise Exception("❌ No model found in Staging or Production stages")

    # 3. LOAD DATA (LOCAL OR S3)
    ref_df = pd.read_csv("data/processed/churn_cleaned.csv")

    s3_bucket = os.getenv('S3_BUCKET')
    s3_key = f"daily/{date}.csv"

    # Try to load from S3 first, fall back to local
    if s3_bucket and os.getenv('AWS_ACCESS_KEY_ID'):
        try:
            logger.info(f"Loading daily data from S3: s3://{s3_bucket}/{s3_key}")
            s3_client = boto3.client('s3')
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            curr_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logger.info(f"✅ Loaded {len(curr_df)} rows from S3")
            data_source = "S3"
        except Exception as e:
            logger.warning(f"Could not load from S3: {e}. Using local file.")
            curr_df = pd.read_csv("data/processed/churn_cleaned.csv")
            data_source = "Local (fallback)"
    else:
        logger.info("No S3 credentials found, using local data file")
        curr_df = pd.read_csv("data/processed/churn_cleaned.csv")
        data_source = "Local"

    # 4. RUN INFERENCE
    logger.info("Running inference...")

    # Prepare features (exclude target and identifier columns)
    features_cols = [col for col in curr_df.columns
                     if col not in ['churn', 'phone number']]
    features = curr_df[features_cols]

    # Get predictions
    raw_predictions = model.predict(features)

    # Handle different prediction formats
    if len(raw_predictions.shape) > 1:
        probabilities = raw_predictions[:, 1]  # Binary classification probabilities
    else:
        probabilities = raw_predictions  # Already probabilities

    # Apply threshold to get binary predictions
    binary_predictions = (probabilities >= threshold).astype(int)

    # Add predictions to current data
    curr_df['prediction'] = binary_predictions
    curr_df['prediction_probability'] = probabilities

    # 5. SAVE PREDICTIONS (LOCAL + S3)
    logger.info("Saving predictions...")

    # Prepare predictions DataFrame with all needed columns
    prediction_cols = ['phone number', 'prediction', 'prediction_probability']

    # Include actual churn if available (for evaluation)
    if 'churn' in curr_df.columns:
        prediction_cols.append('churn')
        has_labels = True
    else:
        has_labels = False

    # Create predictions DataFrame with metadata
    predictions_df = curr_df[prediction_cols].copy()
    predictions_df['inference_timestamp'] = datetime.now().isoformat()
    predictions_df['inference_date'] = date
    predictions_df['model_version'] = model_stage
    predictions_df['data_source'] = data_source

    # Save predictions locally
    local_predictions_path = f"predictions/data/predictions_{date}.csv"
    predictions_df.to_csv(local_predictions_path, index=False)
    logger.info(f"✅ Predictions saved locally: {local_predictions_path}")

    # Create and save metadata
    metadata = {
        'date': date,
        'model_name': model_name,
        'model_stage': model_stage,
        'threshold': threshold,
        'total_predictions': len(predictions_df),
        'positive_predictions': int(predictions_df['prediction'].sum()),
        'positive_rate': float(predictions_df['prediction'].mean()),
        'avg_probability': float(predictions_df['prediction_probability'].mean()),
        'has_actual_labels': has_labels,
        'data_source': data_source,
        'inference_timestamp': datetime.now().isoformat()
    }

    metadata_path = f"predictions/data/metadata_{date}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Upload predictions to S3 if available
    if s3_bucket and os.getenv('AWS_ACCESS_KEY_ID'):
        try:
            s3_client = boto3.client('s3')

            # Upload predictions CSV
            s3_predictions_key = f"predictions/{date}/predictions.csv"
            s3_client.upload_file(local_predictions_path, s3_bucket, s3_predictions_key)
            logger.info(f"✅ Predictions uploaded to S3: s3://{s3_bucket}/{s3_predictions_key}")

            # Upload metadata JSON
            s3_metadata_key = f"predictions/{date}/metadata.json"
            s3_client.upload_file(metadata_path, s3_bucket, s3_metadata_key)

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")

    # 6. DRIFT ANALYSIS
    logger.info("Performing drift analysis...")

    # Add predictions to reference data for fair comparison
    ref_features = ref_df.drop(columns=['churn', 'phone number'], errors='ignore')
    ref_raw_predictions = model.predict(ref_features)

    if len(ref_raw_predictions.shape) > 1:
        ref_probabilities = ref_raw_predictions[:, 1]
    else:
        ref_probabilities = ref_raw_predictions

    ref_df['prediction'] = (ref_probabilities >= threshold).astype(int)

    # Configure column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = 'churn'  # Only in reference data
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = [
        'account length', 'total day minutes', 'total day calls',
        'total eve minutes', 'total eve calls', 'total night minutes',
        'total night calls', 'total intl minutes', 'total intl calls',
        'customer service calls'
    ]
    column_mapping.categorical_features = [
        'state', 'international plan', 'voice mail plan', 'area code'
    ]

    # Create and run drift report
    report = Report(metrics=[
        DataDriftPreset(),
        ColumnDriftMetric(column_name="prediction")
    ])

    report.run(
        reference_data=ref_df,
        current_data=curr_df,
        column_mapping=column_mapping
    )

    # 7. SAVE DRIFT REPORT (LOCAL + S3)
    local_report_path = f"predictions/reports/drift_{date}.html"
    report.save_html(local_report_path)
    logger.info(f"✅ Drift report saved locally: {local_report_path}")

    # Upload drift report to S3 if available
    if s3_bucket and os.getenv('AWS_ACCESS_KEY_ID'):
        try:
            s3_report_key = f"reports/drift_{date}.html"
            s3_client.upload_file(local_report_path, s3_bucket, s3_report_key)
            logger.info(f"✅ Drift report uploaded to S3: s3://{s3_bucket}/{s3_report_key}")
        except Exception as e:
            logger.error(f"Failed to upload drift report to S3: {e}")

    # 8. EXTRACT DRIFT METRICS
    result = report.as_dict()
    drift_share = 0.0
    prediction_drifted = False

    try:
        # Extract DataDriftPreset results
        drift_share = result['metrics'][0]['result']['share_of_drifted_columns']

        # Extract ColumnDriftMetric results for prediction column
        prediction_drifted = result['metrics'][1]['result']['drift_detected']
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing drift results: {e}")
        logger.debug(f"Result structure: {json.dumps(result, indent=2, default=str)}")

    logger.info(f"📊 Drift Share: {drift_share:.3f}")
    logger.info(f"📊 Prediction Drifted: {prediction_drifted}")

    # 9. PUSH METRICS TO PROMETHEUS
    pushgateway_url = os.getenv('PROMETHEUS_PUSHGATEWAY_URL')

    if pushgateway_url:
        try:
            registry = CollectorRegistry()

            # Drift metrics
            g_drift = Gauge('data_drift_share', 'Share of drifted features', registry=registry)
            g_pred_drift = Gauge('prediction_drift_detected', '1 if prediction drifted', registry=registry)
            g_timestamp = Gauge('last_monitoring_timestamp', 'Unix timestamp of last monitoring run', registry=registry)

            # Prediction metrics
            g_total_pred = Gauge('total_predictions', 'Total number of predictions made', registry=registry)
            g_positive_pred = Gauge('positive_predictions', 'Number of positive predictions', registry=registry)
            g_positive_rate = Gauge('prediction_positive_rate', 'Rate of positive predictions', registry=registry)

            # Set metric values
            g_drift.set(float(drift_share))
            g_pred_drift.set(1.0 if prediction_drifted else 0.0)
            g_timestamp.set(datetime.now().timestamp())

            g_total_pred.set(len(predictions_df))
            g_positive_pred.set(int(predictions_df['prediction'].sum()))
            g_positive_rate.set(float(predictions_df['prediction'].mean()))

            # Push to gateway
            push_to_gateway(pushgateway_url, job='batch_prediction', registry=registry)
            logger.info(f"✅ Metrics pushed to Prometheus: {pushgateway_url}")

        except Exception as e:
            logger.warning(f"Could not push to Prometheus: {e}")
    else:
        logger.info("⚠️ No PROMETHEUS_PUSHGATEWAY_URL set, skipping metrics push")

    # 10. DRIFT ALERTS
    if drift_share > 0.3:  # 30% of features drifted
        logger.warning(f"🚨 HIGH DATA DRIFT DETECTED: {drift_share:.3f} (threshold: 0.3)")

    if prediction_drifted:
        logger.warning("🚨 PREDICTION DISTRIBUTION HAS DRIFTED!")

    # 11. RETURN COMPREHENSIVE SUMMARY
    summary = {
        "date": date,
        "status": "success",
        "predictions": {
            "total": len(predictions_df),
            "positive": int(predictions_df['prediction'].sum()),
            "positive_rate": float(predictions_df['prediction'].mean()),
            "avg_probability": float(predictions_df['prediction_probability'].mean()),
            "has_labels": has_labels
        },
        "drift": {
            "drift_share": drift_share,
            "prediction_drifted": prediction_drifted,
            "high_drift_alert": drift_share > 0.3
        },
        "paths": {
            "local_predictions": local_predictions_path,
            "local_metadata": metadata_path,
            "local_report": local_report_path
        },
        "model": {
            "name": model_name,
            "stage": model_stage,
            "threshold": threshold
        },
        "metrics_pushed": pushgateway_url is not None
    }

    # Add S3 URLs if uploaded
    if s3_bucket and os.getenv('AWS_ACCESS_KEY_ID'):
        summary["paths"]["s3_predictions"] = f"s3://{s3_bucket}/predictions/{date}/predictions.csv"
        summary["paths"]["s3_metadata"] = f"s3://{s3_bucket}/predictions/{date}/metadata.json"
        summary["paths"]["s3_report"] = f"s3://{s3_bucket}/reports/drift_{date}.html"

    logger.info(f"✅ Prediction pipeline completed successfully for {date}")
    return summary


if __name__ == "__main__":
    try:
        result = run_prediction()
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        exit(1)