import os
import tarfile

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from scipy.stats import randint  # type: ignore
from six.moves import urllib  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.model_selection import (  # type: ignore
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

# Download the dataset
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Load the dataset
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()

# Train-test split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Create income categories
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# Stratified Shuffle Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove income category to restore the original dataset
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Make a copy of the training set for exploration
housing = strat_train_set.copy()

# Data visualization
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# Debugging steps
print("Column data types:\n", housing.dtypes)
print("First few rows:\n", housing.head())

# Check for NaN values
print("Checking for NaN values:\n", housing.isna().sum())

# Handle only numeric columns for infinite values
housing_num = housing.select_dtypes(include=[np.number])

# Check for infinite values in numeric columns
print("Checking for infinite values in numeric columns:\n", np.isinf(housing_num).sum())

# Drop non-numeric columns for correlation matrix
corr_matrix = housing_num.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Feature engineering
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Prepare data for ML algorithms
housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training
housing_labels = strat_train_set["median_house_value"].copy()

# Handling missing values
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
housing_num_tr = imputer.transform(housing_num)

housing_tr = pd.DataFrame(
    housing_num_tr, columns=housing_num.columns, index=housing.index
)

# Handle categorical attributes
housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Evaluate Linear Regression model
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE:", lin_rmse)

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("Linear Regression MAE:", lin_mae)

# Train Decision Tree model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Evaluate Decision Tree model
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree RMSE:", tree_rmse)

# Simplify the RandomizedSearchCV parameter grid
param_distribs = {
    "n_estimators": randint(low=10, high=100),
    "max_features": randint(low=2, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=5,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Grid Search for Random Forest
param_grid = [
    {"n_estimators": [10, 30, 50], "max_features": [2, 4, 6]},
    {"bootstrap": [False], "n_estimators": [10, 20], "max_features": [2, 3, 4]},
]

grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

# Display best hyperparameters
print("Best params:", grid_search.best_params_)

# Display feature importances
feature_importances = grid_search.best_estimator_.feature_importances_
print(
    "Feature importances:",
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True),
)

# Final model evaluation on test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = pd.DataFrame(
    imputer.transform(X_test_num), columns=X_test_num.columns
)

# Add the additional features to the test set
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

# Make final predictions
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final RMSE on test set:", final_rmse)
