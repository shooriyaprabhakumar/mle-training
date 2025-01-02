import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error


def preprocess_test_data(test_features, model):
    # Use the model's preprocessing from the pipeline
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.transform(test_features)


def score_model(model_folder, data_folder):
    # Load the model
    model_file_path = os.path.join(model_folder, "housing_model.pkl")
    model = joblib.load(model_file_path)

    # Print the model pipeline structure
    print("Model pipeline structure:")
    print(model)

    # Load the test dataset
    test_data_path = os.path.join(data_folder, "housing_data.csv")
    test_data = pd.read_csv(test_data_path)

    # Define the feature columns as they were during training
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

    # Ensure all feature columns exist in the test data
    for col in feature_columns:
        if col not in test_data.columns:
            raise ValueError(f"Column '{col}' is missing from the test data.")

    # Separate features from the test data
    test_features = test_data[feature_columns]

    # Preprocess the test data
    test_num_tr = preprocess_test_data(test_features, model)

    # Debug prints
    print("Processed test data shape:", test_num_tr.shape)

    # Make predictions using the processed features
    predictions = model.predict(test_num_tr)

    # Add predictions to the original test DataFrame
    test_data["predicted_median_house_value"] = predictions

    # Calculate RMSE for logging purposes
    true_values = test_data[
        "median_house_value"
    ]  # Assuming this exists in the test set
    rmse = mean_squared_error(true_values, predictions, squared=False)

    # Start an MLFlow run for scoring
    with mlflow.start_run():
        # Log RMSE metric
        mlflow.log_metric("rmse", rmse)

        # Log the predictions CSV as an artifact
        output_file_path = os.path.join(data_folder, "predictions.csv")
        test_data.to_csv(output_file_path, index=False)
        mlflow.log_artifact(output_file_path)

        print(f"Predictions saved to {output_file_path}")
        print(f"Logged data to MLFlow. RMSE: {rmse}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a model.")
    parser.add_argument(
        "--model-folder", required=True, help="Path to the model folder."
    )
    parser.add_argument("--data-folder", required=True, help="Path to the data folder.")
    args = parser.parse_args()

    score_model(args.model_folder, args.data_folder)
