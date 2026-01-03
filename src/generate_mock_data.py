import pandas as pd
import numpy as np
import os

def generate_mock_data():
    os.makedirs('data', exist_ok=True)
    file_path = 'data/churn.csv'

    # These match the columns your preprocess.py is trying to drop
    data = {
        'state': np.random.choice(['NY', 'CA', 'TX', 'FL'], 100),
        'account length': np.random.randint(1, 200, 100),
        'area code': np.random.choice([415, 408, 510], 100),
        'phone number': [f'{np.random.randint(100,999)}-{np.random.randint(1000,9999)}' for _ in range(100)],
        'international plan': np.random.choice(['yes', 'no'], 100),
        'voice mail plan': np.random.choice(['yes', 'no'], 100),
        'number vmail messages': np.random.randint(0, 50, 100),
        'total day minutes': np.random.uniform(0, 350, 100),
        'total day calls': np.random.randint(0, 150, 100),
        'total day charge': np.random.uniform(0, 60, 100), # DROP TARGET
        'total eve minutes': np.random.uniform(0, 350, 100),
        'total eve calls': np.random.randint(0, 150, 100),
        'total eve charge': np.random.uniform(0, 30, 100),  # DROP TARGET
        'total night minutes': np.random.uniform(0, 350, 100),
        'total night calls': np.random.randint(0, 150, 100),
        'total night charge': np.random.uniform(0, 15, 100), # DROP TARGET
        'total intl minutes': np.random.uniform(0, 20, 100),
        'total intl calls': np.random.randint(0, 10, 100),
        'total intl charge': np.random.uniform(0, 5, 100),  # DROP TARGET
        'customer service calls': np.random.randint(0, 10, 100),
        'churn': np.random.choice([True, False], 100)
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Successfully created mock data with matching schema at {file_path}")

if __name__ == "__main__":
    generate_mock_data()