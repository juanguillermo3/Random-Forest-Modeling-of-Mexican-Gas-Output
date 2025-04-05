
import os
from dotenv import load_dotenv
from datasets import load_oil_production_data
from time_series_utils import lag_predictor_variables  # Make sure to import this properly

# Load environment variables
load_dotenv()
TARGET_VAR = os.getenv("TARGET_VAR")
TEST_RATIO = float(os.getenv("TEST_RATIO"))

# Load the production dataset
prod_data = load_oil_production_data()

# Split index based on the TEST_RATIO
split_index = int(len(prod_data) * (1 - TEST_RATIO))
prod_data["sample"] = ["Train"] * split_index + ["Test"] * (len(prod_data) - split_index)

# Define train/test date index-only DataFrames
training_dates = prod_data[prod_data["sample"] == "Train"][[]]
test_dates = prod_data[prod_data["sample"] == "Test"][[]]

# Define target variable series for training and testing
train_y = prod_data.loc[training_dates.index, TARGET_VAR]
test_y = prod_data.loc[test_dates.index, TARGET_VAR]

# Quality check: Ensure no overlap
assert training_dates.index.max() < test_dates.index.min(), "Date ranges overlap!"

# Print summary
print(f"Training Data: {len(training_dates)} observations from {training_dates.index.min()} to {training_dates.index.max()}")
print(f"Test Data: {len(test_dates)} observations from {test_dates.index.min()} to {test_dates.index.max()}")


def prep_test_and_train(data, forecast_horizon, target_var):
    """
    Generates lagged data for forecasting and splits it into training and testing sets.
    
    Parameters:
    - data (pd.DataFrame): The input dataset containing production data.
    - forecast_horizon (int): The number of steps ahead to forecast.
    - target_var (str): The name of the target variable for forecasting.
    
    Returns:
    - tuple: (train_measurements, test_measurements)
    """
    # Generate lagged predictor variables
    transformed_data = lag_predictor_variables(
        data.copy(), 
        dependent=target_var, 
        lag_orders=[forecast_horizon]
    )[0]  # Only take the first item from the list

    # Use globally defined training and test dates
    train_measurements = training_dates.join(transformed_data, how="inner")
    test_measurements = test_dates.join(transformed_data, how="inner")
    
    return train_measurements, test_measurements

# Exportable variables
__all__ = [
    "TARGET_VAR",
    "TEST_RATIO",
    "training_dates",
    "test_dates",
    "train_y",
    "test_y",
    "prod_data",
    "prep_test_and_train"
]
