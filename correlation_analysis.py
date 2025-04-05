
"""
title: Correlation Analysis
description: 
"""


import pandas as pd

def correlation_analysis(df: pd.DataFrame, dependent: str, independent: list = None):
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
