import pandas as pd
import yaml
import boto3
import os
from datetime import datetime
from io import StringIO, BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_data():
    """
    Fetch data from S3 (production) or use local simulation (development)
    Data engineering team uploads monthly data to S3
    """
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("Starting data fetch process...")

    # Get S3 configuration from params.yaml
    s3_bucket = config['data'].get('s3_source_bucket')
    s3_key = config['data'].get('s3_source_key')
    raw_csv_path = config['data']['raw_csv']

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)

    # Check for S3 configuration and AWS credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Priority: S3 if configured and credentials available
    if s3_bucket and aws_access_key and aws_secret_key:
        try:
            logger.info(f"Fetching data from S3: s3://{s3_bucket}/{s3_key}")

            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )

            # Download the file from S3
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_content = response['Body'].read()

            # Save to local file
            df = pd.read_csv(BytesIO(file_content))
            df.to_csv(raw_csv_path, index=False)

            logger.info(f"✅ Successfully downloaded {len(df)} rows from S3")
            logger.info(f"✅ Data saved to: {raw_csv_path}")

            # Optional: Archive the downloaded file with timestamp
            archive_key = f"archive/churn_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            s3_client.copy_object(
                Bucket=s3_bucket,
                CopySource={'Bucket': s3_bucket, 'Key': s3_key},
                Key=archive_key
            )
            logger.info(f"✅ Source file archived to: s3://{s3_bucket}/{archive_key}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to fetch from S3: {e}")
            logger.warning("Falling back to simulation mode...")
            # Fall through to simulation mode

    # Simulation mode (development/fallback)
    logger.info("Using simulation mode (no S3 configuration found)")
    try:
        # Try to read existing data
        if os.path.exists(raw_csv_path):
            df = pd.read_csv(raw_csv_path)
            logger.info(f"Found existing data with {len(df)} rows")

            # Simulate new monthly data (10% growth)
            new_rows = max(100, int(len(df) * 0.1))
            new_data = df.sample(new_rows, replace=True)

            # Add some randomness to simulate real new data
            new_data = new_data.copy()
            numeric_cols = new_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col not in ['churn']:  # Don't modify target
                    # Add small random variation (±5%)
                    variation = 0.95 + 0.1 * pd.Series(np.random.random(len(new_data)), index=new_data.index)
                    new_data[col] = (new_data[col] * variation).round(2)

            # Combine old and new data
            combined_df = pd.concat([df, new_data], ignore_index=True)
            combined_df.to_csv(raw_csv_path, index=False)

            logger.info(f"✅ Simulated {new_rows} new rows. Total: {len(combined_df)} rows")
        else:
            # Create sample data if file doesn't exist
            logger.warning(f"File {raw_csv_path} not found. Creating sample data...")

            # Create sample churn data
            sample_data = {
                'state': ['CA', 'NY', 'TX', 'FL', 'IL'] * 20,
                'account length': np.random.randint(1, 100, 100),
                'area code': [415, 212, 713, 305, 312] * 20,
                'phone number': [f'555-{i:04d}' for i in range(100)],
                'international plan': ['no', 'yes'] * 50,
                'voice mail plan': ['yes', 'no'] * 50,
                'number vmail messages': np.random.randint(0, 50, 100),
                'total day minutes': np.random.uniform(0, 350, 100),
                'total day calls': np.random.randint(0, 200, 100),
                'total day charge': np.random.uniform(0, 60, 100),
                'total eve minutes': np.random.uniform(0, 350, 100),
                'total eve calls': np.random.randint(0, 200, 100),
                'total eve charge': np.random.uniform(0, 30, 100),
                'total night minutes': np.random.uniform(0, 350, 100),
                'total night calls': np.random.randint(0, 200, 100),
                'total night charge': np.random.uniform(0, 20, 100),
                'total intl minutes': np.random.uniform(0, 20, 100),
                'total intl calls': np.random.randint(0, 20, 100),
                'total intl charge': np.random.uniform(0, 5, 100),
                'customer service calls': np.random.randint(0, 10, 100),
                'churn': np.random.choice(['yes', 'no'], 100, p=[0.15, 0.85])
            }

            df = pd.DataFrame(sample_data)
            df.to_csv(raw_csv_path, index=False)
            logger.info(f"✅ Created sample data with {len(df)} rows")

        return True

    except Exception as e:
        logger.error(f"❌ Simulation mode failed: {e}")
        return False


if __name__ == "__main__":
    import numpy as np

    success = fetch_data()
    exit(0 if success else 1)