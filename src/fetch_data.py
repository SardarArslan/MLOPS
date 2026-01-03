import pandas as pd
import yaml


def fetch_data():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    
    # For now, we'll just read the existing data and "simulate" growth
    df = pd.read_csv(config['data']['raw_csv'])

    # Simulate adding 100 new rows from the "future"
    new_data = df.sample(100)
    updated_df = pd.concat([df, new_data], ignore_index=True)

    # Overwrite the file that DVC tracks
    updated_df.to_csv(config['data']['raw_csv'], index=False)
    print(f"New data fetched! Total rows: {len(updated_df)}")


if __name__ == "__main__":
    fetch_data()