"""
title: Oil Production Dataset
description: Provides access layer, performs trivial transformations to the Oil Dataset.
"""
import os
import pandas as pd

def load_oil_production_data(folder="data", filename="BaseProduccion.xlsx"):
    """Loads and transforms oil production data from an Excel file."""
    file_path = os.path.join(folder, filename)
    
    print("Checking for production data file...")
    
    # Check if file exists
    if filename not in os.listdir(folder):
        raise FileNotFoundError(f"Error: '{filename}' not found in '{folder}' folder.")
    
    print("Production data file found. Loading data...")
    
    # Load production data
    prod_data = pd.read_excel(file_path)
    
    print("Production data successfully loaded.")
    print(f"Columns: {list(prod_data.columns)}")
    print(f"Shape: {prod_data.shape}")
    
    # Print before transformation (3 examples)
    print("Before transformation:")
    print(prod_data["FECHA"].head(3))
    
    print("Recasting time variable as a period variable...")
    # Convert to Period[M]
    prod_data["date"] = pd.to_datetime(prod_data["FECHA"], format="%Y-%m").dt.to_period("M")
    
    # Print after transformation (3 examples)
    print("\nAfter transformation:")
    print(prod_data[["FECHA", "date"]].head(3))
    
    # Sort by date
    prod_data = prod_data.sort_values(by="date")
    
    # Set date as index without dropping it
    prod_data = prod_data.set_index("date", drop=False)
    
    return prod_data
