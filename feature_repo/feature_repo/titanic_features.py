from datetime import datetime, timedelta
from feast import (
    Entity,
    Feature,
    FeatureView,
    FileSource,
    ValueType,
    Field,
    FeatureService,
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

# Define passenger features
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

# Define survival prediction features
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

# Define feature service for survival prediction
survival_prediction_service = FeatureService(
    name="survival_prediction_service",
    features=[
        passenger_features,
        survival_features,
    ],
    description="Features for predicting passenger survival",
)
