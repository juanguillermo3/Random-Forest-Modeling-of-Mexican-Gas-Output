
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, Tuple, List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#
# (0)
#
DEFAULT_RF_PARAMS = {
    "n_estimators": 75,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "n_jobs": -1
}

#
# (1)
#
def train_random_forest(train_df, target_var, hyperparams=None):
    """
    Trains a Random Forest regressor.

    Parameters:
    - train_df (pd.DataFrame): Training data including target variable.
    - target_var (str): Name of the target column.
    - hyperparams (dict): Optional dictionary overriding default hyperparameters.

    Returns:
    - model (RandomForestRegressor): Trained model.
    - features (list): List of feature column names used in training.
    """
    params = DEFAULT_RF_PARAMS.copy()
    if hyperparams:
        params.update(hyperparams)

    features = [col for col in train_df.columns if col != target_var]

    model = RandomForestRegressor(**params)
    model.fit(train_df[features], train_df[target_var])
    return model, features
#
# (2)
#
def evaluate_model(model, test_df, features, target_var):
    """
    Evaluates the model using RMSE and R².

    Returns:
    - dict: Contains RMSE, R², and predictions.
    """
    y_true = test_df[target_var]
    y_pred = model.predict(test_df[features])

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred
    }
#
# (3)
#
def get_feature_importance(model, features):
    """
    Returns a pandas Series of feature importances.
    """
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
#
# (4)
#
import pandas as pd
from typing import Optional
from sklearn.metrics import mean_squared_error, r2_score
#
def get_retraining_sample_from_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prediction_index,
    window_size: int,
    forecast_horizon: int = 1
) -> pd.DataFrame:
    """
    Returns the retraining sample of `window_size` rows that *end*
    `forecast_horizon` steps before the prediction index, using the merged and sorted datasets.
    Ensures there's no data leakage via the target variable.
    
    Parameters:
    - train_df: Training dataframe.
    - test_df: Testing dataframe.
    - prediction_index: Index of the prediction target.
    - window_size: Number of rows in the training window.
    - forecast_horizon: Steps ahead being forecasted (default=1).
    
    Returns:
    - pd.DataFrame: Windowed training sample.
    """
    combined_df = pd.concat([train_df, test_df])
    combined_df = combined_df.sort_index()

    if prediction_index not in combined_df.index:
        raise ValueError(f"Prediction index {prediction_index} not found in combined data.")

    idx_loc = combined_df.index.get_loc(prediction_index)

    if isinstance(idx_loc, slice) or isinstance(idx_loc, list):
        raise ValueError("Prediction index must refer to a single row.")

    end_idx = idx_loc - forecast_horizon
    if end_idx < 0:
        raise ValueError("Not enough data before prediction index to account for forecast horizon.")

    start_idx = max(0, end_idx - window_size)
    return combined_df.iloc[start_idx:end_idx]
#
def evaluate_model_over_sliding_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_var: str,
    forecast_horizon: int,
    window_size: int = 36,
    hyperparams: Optional[dict] = None
) -> dict:
    """
    Evaluates a Random Forest model using a sliding window retraining scheme based on row count,
    avoiding data leakage by adjusting for forecast horizon.

    Parameters:
    - train_df: DataFrame with training data (must include target_var).
    - test_df: DataFrame with testing data (must include target_var).
    - target_var: Name of the target column.
    - window_size: Number of rows (months) in the sliding window (default=36).
    - forecast_horizon: Number of steps ahead being forecasted (default=1).
    - hyperparams: Optional Random Forest hyperparameters.

    Returns:
    - dict: Contains RMSE, R², and full prediction series.
    """
    predictions = []
    y_trues = []
    prediction_dates = []

    combined_df = pd.concat([train_df, test_df])
    combined_df = combined_df.sort_index()

    for prediction_index in test_df.index:
        # Retrieve training window excluding forecast_horizon
        try:
            train_window = get_retraining_sample_from_datasets(
                train_df=train_df,
                test_df=test_df,
                prediction_index=prediction_index,
                window_size=window_size,
                forecast_horizon=forecast_horizon
            )
        except ValueError:
            continue  # Skip if not enough data

        if len(train_window) < window_size:
            continue  # Not enough history to train

        # Train model
        model, features = train_random_forest(train_window, target_var, hyperparams)

        # Extract test point and predict
        test_point = combined_df.loc[[prediction_index]]
        X_test = test_point[features]
        y_pred = model.predict(X_test)[0]
        y_true = test_point[target_var].values[0]

        # Store results
        predictions.append(y_pred)
        y_trues.append(y_true)
        prediction_dates.append(prediction_index)

    y_true_series = pd.Series(y_trues, index=prediction_dates)
    y_pred_series = pd.Series(predictions, index=prediction_dates)

    rmse = mean_squared_error(y_true_series, y_pred_series, squared=False)
    r2 = r2_score(y_true_series, y_pred_series)

    return {
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred_series
    }
