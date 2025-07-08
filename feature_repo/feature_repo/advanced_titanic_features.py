from datetime import datetime, timedelta
from feast import (
    Entity,
    Feature,
    FeatureView,
    FileSource,
    ValueType,
    Field,
    FeatureService,
    RequestFeatureView,
    RequestSource,
    OnDemandFeatureView,
)
from feast.types import Float32, Int64, String
import pandas as pd

# Define the passenger entity
passenger = Entity(
    name="passenger_id",
    value_type=ValueType.INT64,
    description="Passenger ID",
    join_keys=["PassengerId"],
)

# Define the data source for passenger features
passenger_source = FileSource(
    path="data/train_processed.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define basic passenger features
passenger_features = FeatureView(
    name="passenger_features",
    entities=[passenger],
    ttl=timedelta(days=365),
    schema=[
        Field(name="PassengerId", dtype=Int64),
        Field(name="Pclass", dtype=Int64),
        Field(name="Sex", dtype=String),
        Field(name="Age", dtype=Float32),
        Field(name="SibSp", dtype=Int64),
        Field(name="Parch", dtype=Int64),
        Field(name="Fare", dtype=Float32),
        Field(name="Embarked", dtype=String),
        Field(name="Cabin", dtype=String),
        Field(name="Ticket", dtype=String),
        Field(name="Name", dtype=String),
    ],
    source=passenger_source,
    online=True,
)

# Define survival features
survival_features = FeatureView(
    name="survival_features",
    entities=[passenger],
    ttl=timedelta(days=365),
    schema=[
        Field(name="PassengerId", dtype=Int64),
        Field(name="Survived", dtype=Int64),
    ],
    source=passenger_source,
    online=True,
)

# Define request source for on-demand features
request_source = RequestSource(
    name="prediction_request",
    schema=[
        Field(name="PassengerId", dtype=Int64),
        Field(name="current_timestamp", dtype=String),
    ],
)


# Define on-demand feature view for derived features
@OnDemandFeatureView(
    sources=[passenger_features, request_source],
    mode="python",
    schema=[
        Field(name="PassengerId", dtype=Int64),
        Field(name="family_size", dtype=Int64),
        Field(name="is_alone", dtype=Int64),
        Field(name="age_group", dtype=String),
        Field(name="fare_per_person", dtype=Float32),
        Field(name="title", dtype=String),
        Field(name="cabin_deck", dtype=String),
        Field(name="has_cabin", dtype=Int64),
    ],
)
def derived_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from basic passenger data
    """
    df = input_df.copy()

    # Family size
    df["family_size"] = (
        df["passenger_features__SibSp"] + df["passenger_features__Parch"] + 1
    )

    # Is alone
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Age group
    def get_age_group(age):
        if age < 18:
            return "child"
        elif age < 30:
            return "young_adult"
        elif age < 50:
            return "adult"
        else:
            return "senior"

    df["age_group"] = df["passenger_features__Age"].apply(get_age_group)

    # Fare per person
    df["fare_per_person"] = df["passenger_features__Fare"] / df["family_size"]

    # Extract title from name
    def extract_title(name):
        if pd.isna(name):
            return "Unknown"
        return name.split(",")[1].split(".")[0].strip()

    df["title"] = df["passenger_features__Name"].apply(extract_title)

    # Cabin deck
    def get_cabin_deck(cabin):
        if pd.isna(cabin) or cabin == "Unknown":
            return "Unknown"
        return cabin[0] if len(cabin) > 0 else "Unknown"

    df["cabin_deck"] = df["passenger_features__Cabin"].apply(get_cabin_deck)

    # Has cabin
    df["has_cabin"] = (df["passenger_features__Cabin"] != "Unknown").astype(int)

    return df


# Define feature service for advanced survival prediction
advanced_survival_prediction_service = FeatureService(
    name="advanced_survival_prediction_service",
    features=[
        passenger_features,
        survival_features,
        derived_features,
    ],
    description="Advanced features for predicting passenger survival with derived features",
)

# Define feature service for basic prediction
basic_survival_prediction_service = FeatureService(
    name="basic_survival_prediction_service",
    features=[
        passenger_features,
        survival_features,
    ],
    description="Basic features for predicting passenger survival",
)
