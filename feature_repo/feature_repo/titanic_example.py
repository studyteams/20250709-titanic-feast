import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feast import FeatureStore
import warnings

warnings.filterwarnings("ignore")

# Import feature definitions
from titanic_features import (
    passenger,
    passenger_features,
    survival_features,
    survival_prediction_service,
)


def setup_feature_store():
    """
    Set up and configure the feature store
    """
    # Create feature store configuration
    store = FeatureStore(repo_path=".")

    return store


def apply_feature_definitions(store):
    """
    Apply feature definitions to the feature store
    """
    print("Applying feature definitions...")

    # Apply entities
    store.apply([passenger])

    # Apply feature views
    store.apply([passenger_features, survival_features])

    # Apply feature service
    store.apply([survival_prediction_service])

    print("Feature definitions applied successfully!")


def materialize_features(store):
    """
    Materialize features for online serving
    """
    print("Materializing features...")

    # Get the date range for materialization
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Materialize features
    store.materialize(start_date=start_date, end_date=end_date)

    print("Features materialized successfully!")


def get_training_data(store):
    """
    Get training data with features for model training
    """
    print("Getting training data...")

    # Read the training data
    train_df = pd.read_parquet("data/train_processed.parquet")

    # Get features for training
    entity_df = train_df[["PassengerId", "event_timestamp"]].copy()

    # Ensure event_timestamp is datetime
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # Get historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
            "survival_features:Survived",
        ],
        full_feature_names=True,
    ).to_df()

    print(f"Training data shape: {training_df.shape}")
    return training_df


def train_model(training_df):
    """
    Train a simple machine learning model
    """
    print("Training model...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report

    # Prepare features
    feature_columns = [
        "passenger_features__Pclass",
        "passenger_features__Sex",
        "passenger_features__Age",
        "passenger_features__SibSp",
        "passenger_features__Parch",
        "passenger_features__Fare",
        "passenger_features__Embarked",
    ]

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    X = training_df[feature_columns].copy()
    X["passenger_features__Sex"] = le_sex.fit_transform(X["passenger_features__Sex"])
    X["passenger_features__Embarked"] = le_embarked.fit_transform(
        X["passenger_features__Embarked"]
    )

    y = training_df["survival_features__Survived"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, le_sex, le_embarked


def get_online_features(store, passenger_ids):
    """
    Get features for online prediction
    """
    print("Getting online features...")

    # Create entity dataframe for online features
    entity_df = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "event_timestamp": [datetime.now()] * len(passenger_ids),
        }
    )

    # Get online features
    online_features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
        ],
        entity_rows=[{"PassengerId": pid} for pid in passenger_ids],
    ).to_df()

    return online_features


def predict_survival(store, model, le_sex, le_embarked, passenger_ids):
    """
    Predict survival for given passenger IDs
    """
    print("Making predictions...")

    # Get online features
    online_features = get_online_features(store, passenger_ids)

    # Debug: print available columns
    print("Available columns:", list(online_features.columns))

    # Prepare features for prediction - use simple column names for online features
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    # Check which columns are actually available
    available_columns = [
        col for col in feature_columns if col in online_features.columns
    ]
    print("Available feature columns:", available_columns)

    if not available_columns:
        print(
            "No feature columns found. Using all available columns except PassengerId."
        )
        available_columns = [
            col for col in online_features.columns if col != "PassengerId"
        ]

    X_pred = online_features[available_columns].copy()

    # Fill missing values
    fill_values = {}
    if "Age" in X_pred.columns:
        fill_values["Age"] = X_pred["Age"].median()
    if "Fare" in X_pred.columns:
        fill_values["Fare"] = X_pred["Fare"].median()
    if "Sex" in X_pred.columns:
        fill_values["Sex"] = "male"  # Default value for missing sex
    if "Embarked" in X_pred.columns:
        fill_values["Embarked"] = "S"  # Default value for missing embarked
    if "Pclass" in X_pred.columns:
        fill_values["Pclass"] = 3  # Default value for missing pclass
    if "SibSp" in X_pred.columns:
        fill_values["SibSp"] = 0  # Default value for missing sibsp
    if "Parch" in X_pred.columns:
        fill_values["Parch"] = 0  # Default value for missing parch

    X_pred = X_pred.fillna(fill_values)

    # Encode categorical variables using simple column names
    if "Sex" in X_pred.columns:
        X_pred["Sex"] = le_sex.transform(X_pred["Sex"])
    if "Embarked" in X_pred.columns:
        X_pred["Embarked"] = le_embarked.transform(X_pred["Embarked"])

    # Rename columns to match training data format
    column_mapping = {
        "Pclass": "passenger_features__Pclass",
        "Sex": "passenger_features__Sex",
        "Age": "passenger_features__Age",
        "SibSp": "passenger_features__SibSp",
        "Parch": "passenger_features__Parch",
        "Fare": "passenger_features__Fare",
        "Embarked": "passenger_features__Embarked",
    }

    X_pred = X_pred.rename(columns=column_mapping)

    # Make predictions
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)

    # Create results dataframe
    results = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Survived_Prediction": predictions,
            "Survival_Probability": probabilities[:, 1],
        }
    )

    return results


def main():
    """
    Main function to run the complete Titanic example
    """
    print("=== Titanic Survival Prediction with Feast ===\n")

    # Step 1: Set up feature store
    print("1. Setting up feature store...")
    store = setup_feature_store()

    # Step 2: Apply feature definitions
    print("\n2. Applying feature definitions...")
    apply_feature_definitions(store)

    # Step 3: Materialize features
    print("\n3. Materializing features...")
    materialize_features(store)

    # Step 4: Get training data
    print("\n4. Getting training data...")
    training_df = get_training_data(store)

    # Step 5: Train model
    print("\n5. Training model...")
    model, le_sex, le_embarked = train_model(training_df)

    # Step 6: Make predictions on test data
    print("\n6. Making predictions on test data...")
    test_df = pd.read_parquet("data/test_processed.parquet")
    test_passenger_ids = (
        test_df["PassengerId"].head(10).tolist()
    )  # Predict for first 10 passengers

    predictions = predict_survival(
        store, model, le_sex, le_embarked, test_passenger_ids
    )

    print("\nPrediction Results:")
    print(predictions)

    # Step 7: Show feature importance
    print("\n7. Feature Importance:")
    feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print(importance_df)

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
