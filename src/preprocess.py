import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml
import os


def preprocess():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data']['raw_csv'])

    # 1. Drop redundant columns
    cols_to_drop = ['total day charge', 'total eve charge',
                    'total night charge', 'total intl charge']
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. Encoding
    binary_map = {'yes': 1, 'no': 0}
    df['international plan'] = df['international plan'].map(binary_map)
    df['voice mail plan'] = df['voice mail plan'].map(binary_map)
    df['churn'] = df['churn'].astype(int)

    # 3. Handle Area Code & State
    le = LabelEncoder()
    df['state'] = le.fit_transform(df['state'])
    df['area code'] = le.fit_transform(df['area code'].astype(str))

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/churn_cleaned.csv", index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()