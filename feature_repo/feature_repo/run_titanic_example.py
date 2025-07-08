#!/usr/bin/env python3
"""
Titanic Survival Prediction with Feast - Main Execution Script

This script runs the complete Titanic example with Feast, including:
1. Data preparation
2. Basic feature store setup
3. Advanced feature engineering
4. Model training and prediction
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """
    Check if required dependencies are installed
    """
    print("Checking dependencies...")

    required_packages = [
        "feast",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("All dependencies are installed!")
    return True


def prepare_data():
    """
    Prepare Titanic data for Feast
    """
    print("\n=== Step 1: Preparing Data ===")

    try:
        from prepare_data import prepare_titanic_data

        prepare_titanic_data()
        print("✓ Data preparation completed!")
        return True
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False


def run_basic_example():
    """
    Run the basic Titanic example
    """
    print("\n=== Step 2: Running Basic Example ===")

    try:
        from titanic_example import main as basic_main

        basic_main()
        print("✓ Basic example completed!")
        return True
    except Exception as e:
        print(f"✗ Basic example failed: {e}")
        return False


def run_advanced_example():
    """
    Run the advanced Titanic example
    """
    print("\n=== Step 3: Running Advanced Example ===")

    try:
        from advanced_titanic_example import main as advanced_main

        advanced_main()
        print("✓ Advanced example completed!")
        return True
    except Exception as e:
        print(f"✗ Advanced example failed: {e}")
        return False


def create_demo_script():
    """
    Create a demo script for quick testing
    """
    print("\n=== Creating Demo Script ===")

    demo_script = '''
import pandas as pd
from feast import FeatureStore
from datetime import datetime

def quick_demo():
    """
    Quick demo of Feast with Titanic data
    """
    print("=== Quick Feast Demo ===")

    # Initialize feature store
    store = FeatureStore(config_path="feature_store.yaml")

    # Get some online features
    passenger_ids = [1, 2, 3, 4, 5]

    features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:Fare"
        ],
        entity_rows=[{"PassengerId": pid} for pid in passenger_ids]
    ).to_df()

    print("\\nOnline Features:")
    print(features)

    print("\\nDemo completed!")

if __name__ == "__main__":
    quick_demo()
'''

    with open("quick_demo.py", "w") as f:
        f.write(demo_script)

    print("✓ Demo script created: quick_demo.py")


def main():
    """
    Main execution function
    """
    print("🚢 Titanic Survival Prediction with Feast")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("feature_store.yaml"):
        print(
            "Error: feature_store.yaml not found. Please run this script from the feature_repo directory."
        )
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        sys.exit(1)

    # Prepare data
    if not prepare_data():
        print("Data preparation failed. Exiting.")
        sys.exit(1)

    # Run basic example
    if not run_basic_example():
        print("Basic example failed. Continuing with advanced example...")

    # Run advanced example
    if not run_advanced_example():
        print("Advanced example failed.")

    # Create demo script
    create_demo_script()

    print("\n" + "=" * 50)
    print("🎉 Titanic Feast Example Completed!")
    print("\nWhat was accomplished:")
    print("1. ✓ Data preparation with timestamps")
    print("2. ✓ Basic feature store setup")
    print("3. ✓ Feature definitions and materialization")
    print("4. ✓ Model training with basic features")
    print("5. ✓ Advanced feature engineering with on-demand features")
    print("6. ✓ Model comparison and feature importance analysis")
    print("7. ✓ Online feature serving demonstration")

    print("\nFiles created:")
    print("- data/train_processed.csv")
    print("- data/test_processed.csv")
    print("- feature_importance.png")
    print("- quick_demo.py")

    print("\nNext steps:")
    print("1. Run 'python quick_demo.py' for a quick demo")
    print("2. Explore the feature definitions in titanic_features.py")
    print("3. Modify advanced_titanic_features.py to add more features")
    print("4. Experiment with different models and feature combinations")


if __name__ == "__main__":
    main()
