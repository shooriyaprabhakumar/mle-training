import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def train_model(data_folder):
    """
    Train a RandomForest model on the housing data and save the model.

    Parameters
    ----------
    data_folder : str
        The folder containing the training data CSV file.

    Returns
    -------
    None
    """
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
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("cat", OneHotEncoder(), ["ocean_proximity"]),
        ]
    )

    # Create a pipeline with preprocessing and the model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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
