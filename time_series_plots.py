
"""
title: Time Series Plots
description: Provides graphical tools based on Matplotlib to visualize time series. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_time_series(
    data, time_col, value_col, sample_col=None, style="whitegrid"
):
    """
    Converts a time column to PeriodIndex, then to Timestamp, and plots the time series
    using a line plot. Optionally, segments the plot based on a sample column,
    changing colors for different samples and adding vertical dashed lines at sample boundaries.

    Parameters:
    - data (pd.DataFrame): Input DataFrame
    - time_col (str): Column name containing time information (YYYY-MM format)
    - value_col (str): Column name containing values to plot
    - sample_col (str, optional): Column name indicating different samples (segments)
    - style (str): Seaborn style for visualization (default: "whitegrid")
    """
    sns.set_theme(style=style)
    df = data.copy()

    # Convert time column to PeriodIndex and then to Timestamp
    df["date"] = pd.to_datetime(df[time_col], format="%Y-%m").dt.to_period("M")
    df.set_index("date", inplace=True)
    df.index = df.index.to_timestamp()

    plt.figure(figsize=(10, 5))

    if sample_col and sample_col in df.columns:
        unique_samples = df[sample_col].unique()
        palette = sns.color_palette("tab10", len(unique_samples))
        sample_colors = dict(zip(unique_samples, palette))

        previous_sample = None
        for sample in unique_samples:
            segment = df[df[sample_col] == sample]
            plt.plot(
                segment.index,
                segment[value_col],
                linestyle="-",
                #marker="o",
                color=sample_colors[sample],
                label=str(sample),
                alpha=0.85,
            )

            # Draw vertical line at the start of this segment (except the first one)
            if previous_sample is not None:
                boundary_date = segment.index.min()
                plt.axvline(boundary_date, color="gray", linestyle="--", linewidth=1.4, alpha=0.8)

            previous_sample = sample
    else:
        plt.plot(
            df.index,
            df[value_col],
            linestyle="-",
            color="b",
            linewidth=1.5,
            label=value_col,
            alpha=0.85,
        )

    plt.xlabel("Time (Months)")
    plt.ylabel(value_col)
    plt.title(f"Time Series Plot ({value_col} over {time_col})")
    plt.grid(True)
    plt.legend()
    plt.show()
