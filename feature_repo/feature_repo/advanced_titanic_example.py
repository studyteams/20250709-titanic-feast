import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feast import FeatureStore
import warnings

warnings.filterwarnings("ignore")

# Import advanced feature definitions
from advanced_titanic_features import (
    passenger,
    passenger_features,
    survival_features,
    derived_features,
    advanced_survival_prediction_service,
    basic_survival_prediction_service,
    request_source,
)


def setup_feature_store():
    """
    Set up and configure the feature store
    """
    store = FeatureStore(repo_path=".")
    return store


def apply_advanced_feature_definitions(store):
    """
    Apply advanced feature definitions to the feature store
    """
    print("Applying advanced feature definitions...")

    # Apply entities
    store.apply([passenger])

    # Apply feature views
    store.apply([passenger_features, survival_features])

    # Apply request source
    store.apply([request_source])

    # Apply on-demand feature view
    store.apply([derived_features])

    # Apply feature services
    store.apply(
        [advanced_survival_prediction_service, basic_survival_prediction_service]
    )

    print("Advanced feature definitions applied successfully!")


def get_advanced_training_data(store):
    """
    Get training data with advanced features for model training
    """
    print("Getting advanced training data...")

    # Read the training data
    train_df = pd.read_parquet("data/train_processed.parquet")

    # Get features for training
    entity_df = train_df[["PassengerId", "event_timestamp"]].copy()

    # Ensure event_timestamp is datetime
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # Add request data for on-demand features
    request_df = pd.DataFrame(
        {
            "PassengerId": train_df["PassengerId"],
            "current_timestamp": [datetime.now().isoformat()] * len(train_df),
        }
    )

    # Get historical features with on-demand features
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
            "passenger_features:Cabin",
            "passenger_features:Name",
            "derived_features:family_size",
            "derived_features:is_alone",
            "derived_features:age_group",
            "derived_features:fare_per_person",
            "derived_features:title",
            "derived_features:cabin_deck",
            "derived_features:has_cabin",
            "survival_features:Survived",
        ],
        full_feature_names=True,
    ).to_df()

    print(f"Advanced training data shape: {training_df.shape}")
    return training_df


def train_advanced_model(training_df):
    """
    Train an advanced machine learning model with derived features
    """
    print("Training advanced model...")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare features
    basic_features = [
        "passenger_features__Pclass",
        "passenger_features__Sex",
        "passenger_features__Age",
        "passenger_features__SibSp",
        "passenger_features__Parch",
        "passenger_features__Fare",
        "passenger_features__Embarked",
    ]

    derived_feature_columns = [
        "derived_features__family_size",
        "derived_features__is_alone",
        "derived_features__age_group",
        "derived_features__fare_per_person",
        "derived_features__title",
        "derived_features__cabin_deck",
        "derived_features__has_cabin",
    ]

    all_features = basic_features + derived_feature_columns

    # Encode categorical variables
    categorical_features = [
        "passenger_features__Sex",
        "passenger_features__Embarked",
        "derived_features__age_group",
        "derived_features__title",
        "derived_features__cabin_deck",
    ]

    encoders = {}
    X = training_df[all_features].copy()

    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        encoders[feature] = le

    y = training_df["survival_features__Survived"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train multiple models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    print(
        f"\nBest model: {best_model.__class__.__name__} with accuracy: {best_score:.4f}"
    )

    # Feature importance analysis
    feature_names = [
        f.replace("passenger_features__", "").replace("derived_features__", "")
        for f in all_features
    ]

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature")
    plt.title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    return best_model, encoders, all_features


def predict_with_advanced_features(
    store, model, encoders, feature_columns, passenger_ids
):
    """
    Predict survival using advanced features
    """
    print("Making predictions with advanced features...")

    # Create request data
    request_data = [
        {"PassengerId": pid, "current_timestamp": datetime.now().isoformat()}
        for pid in passenger_ids
    ]

    # Get online features with on-demand features
    online_features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
            "passenger_features:Cabin",
            "passenger_features:Name",
            "derived_features:family_size",
            "derived_features:is_alone",
            "derived_features:age_group",
            "derived_features:fare_per_person",
            "derived_features:title",
            "derived_features:cabin_deck",
            "derived_features:has_cabin",
        ],
        entity_rows=[{"PassengerId": pid} for pid in passenger_ids],
        request_data=request_data,
    ).to_df()

    # Prepare features for prediction
    X_pred = online_features[feature_columns].copy()

    # Encode categorical variables
    categorical_features = [
        "passenger_features__Sex",
        "passenger_features__Embarked",
        "derived_features__age_group",
        "derived_features__title",
        "derived_features__cabin_deck",
    ]

    for feature in categorical_features:
        if feature in encoders:
            X_pred[feature] = encoders[feature].transform(X_pred[feature].astype(str))

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


def compare_basic_vs_advanced(store):
    """
    Compare basic vs advanced feature sets
    """
    print("\n=== Comparing Basic vs Advanced Features ===")

    # Get test data
    test_df = pd.read_parquet("data/test_processed.parquet")
    test_passenger_ids = test_df["PassengerId"].head(5).tolist()

    # Basic features prediction
    print("\n1. Basic Features Prediction:")
    basic_features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
        ],
        entity_rows=[{"PassengerId": pid} for pid in test_passenger_ids],
    ).to_df()

    print("Basic features shape:", basic_features.shape)
    print("Basic features columns:", list(basic_features.columns))

    # Advanced features prediction
    print("\n2. Advanced Features Prediction:")
    request_data = [
        {"PassengerId": pid, "current_timestamp": datetime.now().isoformat()}
        for pid in test_passenger_ids
    ]

    advanced_features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
            "derived_features:family_size",
            "derived_features:is_alone",
            "derived_features:age_group",
            "derived_features:fare_per_person",
            "derived_features:title",
            "derived_features:cabin_deck",
            "derived_features:has_cabin",
        ],
        entity_rows=[{"PassengerId": pid} for pid in test_passenger_ids],
        request_data=request_data,
    ).to_df()

    print("Advanced features shape:", advanced_features.shape)
    print("Advanced features columns:", list(advanced_features.columns))

    print("\nFeature comparison completed!")


def main():
    """
    Main function to run the advanced Titanic example
    """
    print("=== Advanced Titanic Survival Prediction with Feast ===\n")

    # Step 1: Set up feature store
    print("1. Setting up feature store...")
    store = setup_feature_store()

    # Step 2: Apply advanced feature definitions
    print("\n2. Applying advanced feature definitions...")
    apply_advanced_feature_definitions(store)

    # Step 3: Materialize features
    print("\n3. Materializing features...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    store.materialize(start_date=start_date, end_date=end_date)

    # Step 4: Get advanced training data
    print("\n4. Getting advanced training data...")
    training_df = get_advanced_training_data(store)

    # Step 5: Train advanced model
    print("\n5. Training advanced model...")
    model, encoders, feature_columns = train_advanced_model(training_df)

    # Step 6: Make predictions with advanced features
    print("\n6. Making predictions with advanced features...")
    test_df = pd.read_parquet("data/test_processed.parquet")
    test_passenger_ids = test_df["PassengerId"].head(10).tolist()

    predictions = predict_with_advanced_features(
        store, model, encoders, feature_columns, test_passenger_ids
    )

    print("\nAdvanced Prediction Results:")
    print(predictions)

    # Step 7: Compare basic vs advanced features
    print("\n7. Comparing basic vs advanced features...")
    compare_basic_vs_advanced(store)

    print("\n=== Advanced Example completed successfully! ===")
    print("Feature importance plot saved as 'feature_importance.png'")


if __name__ == "__main__":
    main()
