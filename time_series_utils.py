
import pandas as pd

def lag_predictor_variables(df: pd.DataFrame, dependent: str, lag_orders: list):
    """
    Generates multiple lagged predictor datasets for different lag orders while keeping the original index
    and dependent variable unchanged.
    
    Parameters:
        df (pd.DataFrame): The input dataset with time-series data.
        dependent (str): The target variable (remains unchanged).
        lag_orders (list): A list of integers specifying the lag orders for predictors.
    
    Returns:
        list of pd.DataFrame: A list of DataFrames, each corresponding to a different lag order.
    """
    results = []
    
    for lag in lag_orders:
        df_lagged = df.copy()
        df_lagged[df.drop(columns=[dependent]).columns] = df.drop(columns=[dependent]).shift(lag)  # Shift predictors
        df_lagged = df_lagged.dropna()  # Drop NaN rows caused by shifting
        results.append(df_lagged)  # Append to the results list
    
    return results
