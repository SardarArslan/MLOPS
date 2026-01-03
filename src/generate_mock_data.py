import pandas as pd
import numpy as np
import os


def generate_mock_data():
    # Define the directory and file path
    os.makedirs('data', exist_ok=True)
    file_path = 'data/churn.csv'

    # Create 100 rows of fake telecom churn data
    data = {
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(100, 8000, 100),
        'Churn': np.random.choice(['Yes', 'No'], 100)
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Successfully created mock data at {file_path}")


if __name__ == "__main__":
    generate_mock_data()