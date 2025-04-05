"""
title: Feature Names Tools
description: Simplifies feature names.
"""
import pandas as pd

def rename_columns_for_us_audience(df: pd.DataFrame) -> pd.DataFrame:
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
