import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Custom transformer for additional features
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def train_model(data_folder):
    # Load the training data
    data_path = os.path.join(data_folder, "housing_data.csv")
    data = pd.read_csv(data_path)

    # Define features and target
    feature_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]
    target_column = "median_house_value"

    X = data[feature_columns]
    y = data[target_column]

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the preprocessing steps
    num_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), ["ocean_proximity"]),
        ]
    )

    # Create a pipeline with preprocessing, feature addition, and the model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_adder", FeatureAdder()),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Save the model
    model_file_path = os.path.join(data_folder, "housing_model.pkl")
    joblib.dump(model, model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--data-folder", required=True, help="Path to the data folder.")
    args = parser.parse_args()

    train_model(args.data_folder)
