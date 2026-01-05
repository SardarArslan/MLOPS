# MLOps Pipeline: Telecom Churn Prediction & Monitoring
A complete, production-grade MLOps pipeline for predicting customer churn. This system automates model training, prediction, drift monitoring, and visualization, implementing industry best practices for maintainable machine learning operations.

## Features
- **Automated Model Lifecycle**: Scheduled retraining and daily inference via GitHub Actions

- **Production Monitoring**: Real-time dashboards for data drift, prediction quality, and system health

- **Reproducible Pipelines**: Versioned data and models with DVC, tracked experiments with MLflow

## Architecture
| Phase | Component | Input | Tooling | Schedule/Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Data Sources | S3 / Local Storage | — | Monthly |
| **Training** | Training Pipeline | Raw Data | Python / Scripts | Monthly (Cron) |
| **Training** | Model Registry | Trained Models | MLflow | Continuous |
| **Inference** | Daily Data | S3 / Local Storage | — | Daily |
| **Inference** | Prediction | Daily Data + Model | Prediction Pipeline | Daily (Cron) |
| **Monitoring** | Metrics Collection | Prediction Results | Prometheus | Real-time |
| **Monitoring** | Dashboard | Performance Metrics | Grafana | Real-time |
| **Monitoring** | Alerting | System Health | Visualizations | Automated |



#### 📁 Project Structure

```text
mlops-telecom-churn/
├── src/
│   ├── fetch_data.py      # Fetches data from S3 with local fallback
│   ├── preprocess.py      # Data cleaning and feature engineering
│   ├── train.py           # Model training and MLflow logging
│   └── predict.py         # Batch prediction, drift analysis, metrics export
├── data/
│   ├── raw/               # Raw data (DVC-tracked)
│   └── processed/         # Cleaned features
├── predictions/
│   ├── data/              # Daily prediction outputs
│   └── reports/           # Evidently.ai drift reports
├── grafana/
│   ├── dashboards/
│   └── provisioning/      # Automated Grafana configuration and dashboard json
├── mlruns/                # MLflow experiment tracking
├── docker-compose.monitoring.yml  # Monitoring stack
├── dvc.yaml               # Data versioning pipeline
├── params.yaml            # All configuration parameters
└── .github/workflows/     # CI/CD pipelines
    ├── retrain.yml        # Monthly model retraining
    └── predict.yml        # Daily prediction & monitoring
```
## Quickstart (mac local)
- brew install docker docker-compose
- curl -LsSf https://astral.sh/uv/install.sh | sh
- git clone https://github.com/SardarArslan/MLOPS.git
- cd MLOPS
- uv sync
- touch .env (Check .env.template)
- dvc init
- mkdir tmp/dvcstore
- dvc remote add tmp/dvcstore (or s3 bucket)
- mlflow ui --backend-store-uri sqlite:///mlflow.db
- uv run dvc repro
- docker-compose -f docker-compose.monitoring.yml up -d (monitoring setup)
- uv run python src/predict.py
- Check Grafana: http://localhost:3000 (admin/admin)
- Check Prometheus: http://localhost:9090
- Check Pushgateway: http://localhost:9091
- It will auto-load the dashboard in grafana

