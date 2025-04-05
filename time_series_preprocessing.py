
import pandas as pd
def add_lagged_features(df, target_var, back_sight=1):
    """
    Adds lagged features up to `back_sight` steps for all columns except the target variable.

    Parameters:
    - df (pd.DataFrame): Date-indexed dataframe with input features and a target variable.
    - target_var (str): Name of the column to use as the target variable.
    - back_sight (int): Number of lag periods to include as features.

    Returns:
    - pd.DataFrame: The dataframe with lagged features added (and NaNs dropped).
    - str: The name of the target variable.
    - int: The back_sight value.
    """
    df = df.copy()
    non_target_cols = [col for col in df.columns if col != target_var]

    for col in non_target_cols:
        for lag in range(1, back_sight + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Drop rows with any NaN values introduced by lagging
    df = df.dropna()

    return df
