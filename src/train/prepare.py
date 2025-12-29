import pickle
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer

# Make sure we can import from util
import sys


def read_data(filename):
    df = pd.read_parquet(filename)
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)
    df = df[(df.duration>=1) & (df.duration<=60)].reset_index(drop=True)
    categorical = ["PULocationID","DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def prepare():
    """Prepare and save processed data"""

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Read data
    print("Reading data...")
    df_train = read_data('data/green_tripdata_2025-01.parquet')
    df_val = read_data('data/green_tripdata_2025-02.parquet')

    # Feature engineering
    print("Engineering features...")
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    # Prepare features
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]


    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    # Save processed data
    print("Saving processed data...")
    with open("data/processed/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open("data/processed/X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)

    with open("data/processed/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    with open("data/processed/y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)

    with open("data/processed/dv.pkl", "wb") as f:
        pickle.dump(dv, f)

    # Calculate and save data statistics
    stats = {
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "train_mean_duration": float(y_train.mean()),
        "train_std_duration": float(y_train.std()),
        "val_mean_duration": float(y_val.mean()),
        "val_std_duration": float(y_val.std()),
        "features_count": X_train.shape[1]
    }

    with open("data/processed/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Preparation complete!")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['val_samples']}")
    print(f"Features: {stats['features_count']}")


if __name__ == "__main__":
    prepare()