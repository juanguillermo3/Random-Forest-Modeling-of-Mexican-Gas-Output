
"""
title: Temporal correlations
description: Provides several tools for a correlation analysis in a multivariate time series contexts. 
"""

import pandas as pd
import numpy as np

def safe_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a dataframe by applying first differencing and standardization.
    - Fails fast if any feature is non-numeric.
    - Prints descriptive statistics before and after transformation.
    
    Args:
        df (pd.DataFrame): Dataframe containing numerical measurements.
    
    Returns:
        pd.DataFrame: Transformed dataframe in standardized first-difference space.
    """
    # 🚨 Fail fast if any column is non-numeric
    if not np.all(np.issubdtype(df[col].dtype, np.number) for col in df.columns):
        raise ValueError("All columns must be numeric for safe_transform to proceed.")
    
    print("\n📊 Before Transformation:")
    print(df.describe())

    # Compute first differences (ΔX = X_t - X_(t-1))
    df_diff = df.diff().dropna()  # Drop first row with NaNs
    
    # Standardization (Z-score normalization)
    df_standardized = (df_diff - df_diff.mean()) / df_diff.std()

    print("\n📈 After Transformation:")
    print(df_standardized.describe())

    return df_standardized

import pandas as pd

def compute_correlations(df: pd.DataFrame, dependent: str, independent: list = None):
    """
    Computes Pearson correlation coefficients between a dependent variable and multiple independent variables.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing numerical data.
        dependent (str): The name of the dependent variable.
        independent (list, optional): List of independent variable names. If None, use all numerical except dependent.

    Returns:
        dict: A dictionary where keys are independent variables and values are their correlation coefficients.
    """
    # Auto-select independent variables if not provided
    if independent is None:
        independent = [col for col in df.select_dtypes(include=[float, int]).columns if col != dependent]

    return {var: df[dependent].corr(df[var]) for var in independent}
