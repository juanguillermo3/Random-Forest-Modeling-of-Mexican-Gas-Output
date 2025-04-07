"""
title: Gas Production Dataset
description: Provides a data access layer for Mexico's Gas Dataset — a comprehensive collection of data on national
             natural gas production, along with various socioeconomic and industrial indicators.
"""

#
# (0)
#
import os
import pandas as pd
#
# (1)
#
def load_gas_production_data(folder="data", filename="BaseProduccion.xlsx"):
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
#
# (2)
#
def translate_gas_dataset_for_us_audience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to be more understandable for an American audience,
    simplifying names and using underscores instead of spaces.
    """
    rename_map = {
        "FECHA": "date",
        "PRODUCCION NACIONAL (GAS MMPCD)": "national_production",
        "Impotacion  (GAS MMPCD)": "importation",
        "Demanda  (GAS MMPCD)": "demand",
        "PRECIO (Dollars per MWh)": "price",
        "CONSUMO(terawatt-hours.)": "consumption",
        "1P (MMpc)": "proven_reserves_1p",
        "2P (MMpc)": "probable_reserves_2p",
        "3P (MMpc)": "possible_reserves_3p",
        "Pozos": "wells",
        "Sector_Petrolero (GAS MMPCD)": "oil_sector",
        "Sector_Industrial (GAS MMPCD)": "industrial_sector",
        "Sectores_Residencial_Servicios_Autotransporte (GAS MMPCD)": "residential_services_transport",
        "Sector_Electrico (GAS MMPCD)": "electric_sector",
        "Poblacion": "population"
    }
    
    df_renamed = df.rename(columns=rename_map)
    return df_renamed
