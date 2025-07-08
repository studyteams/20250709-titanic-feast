import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def prepare_titanic_data():
    """
    Prepare Titanic data for Feast by adding timestamp columns
    """
    # Read the original data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Create timestamp columns
    # Use a base date and add random days to simulate different event times
    base_date = datetime(1912, 4, 10)  # Titanic departure date

    # For training data, create timestamps
    np.random.seed(42)  # For reproducibility
    random_days = np.random.randint(0, 30, len(train_df))
    train_df["event_timestamp"] = pd.to_datetime([
        base_date + timedelta(days=int(days)) for days in random_days
    ])
    train_df["created_timestamp"] = train_df["event_timestamp"]

    # For test data, create timestamps
    random_days_test = np.random.randint(0, 30, len(test_df))
    test_df["event_timestamp"] = pd.to_datetime([
        base_date + timedelta(days=int(days)) for days in random_days_test
    ])
    test_df["created_timestamp"] = test_df["event_timestamp"]

    # Fill missing values
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    train_df["Embarked"] = train_df["Embarked"].fillna("S")
    train_df["Cabin"] = train_df["Cabin"].fillna("Unknown")

    test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
    test_df["Embarked"] = test_df["Embarked"].fillna("S")
    test_df["Cabin"] = test_df["Cabin"].fillna("Unknown")
    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

    # Save processed data as CSV
    train_df.to_csv("data/train_processed.csv", index=False)
    test_df.to_csv("data/test_processed.csv", index=False)

    # Save processed data as Parquet
    train_df.to_parquet("data/train_processed.parquet", index=False)
    test_df.to_parquet("data/test_processed.parquet", index=False)

    print(f"Processed training data: {len(train_df)} rows")
    print(f"Processed test data: {len(test_df)} rows")
    print(
        "Data saved to data/train_processed.csv, data/test_processed.csv, data/train_processed.parquet, and data/test_processed.parquet"
    )


if __name__ == "__main__":
    prepare_titanic_data()
