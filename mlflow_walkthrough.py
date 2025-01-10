import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set parameters for the model
n_estimators = 100
max_depth = 10
random_state = 42

# Start an MLFlow experiment
with mlflow.start_run():
    # Train a RandomForest model
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    predictions = model.predict(X_valid)

    # Calculate metrics
    rmse = mean_squared_error(y_valid, predictions, squared=False)
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log the trained model
    mlflow.sklearn.log_model(model, "model")

    print(f"Logged data to MLFlow. RMSE: {rmse}, MAE: {mae}, R²: {r2}")
