
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(test_dates_df, y_test, evaluation_results):
    """
    Creates one subplot per forecast horizon, showing actual vs. predicted values.

    Parameters:
    - test_dates_df (pd.DataFrame): Index-only DataFrame with a PeriodIndex (monthly).
    - y_test (array-like): True values to compare against predictions (same order as index).
    - evaluation_results (list of dicts): Each dict must include:
        - 'horizon': forecast horizon in months,
        - 'predictions': array-like, same length as y_test,
        - optionally 'r2': R-squared value of the predictions.
    """
    if not isinstance(test_dates_df.index, pd.PeriodIndex):
        raise ValueError("`test_dates_df` must have a PeriodIndex with monthly frequency.")

    test_index = test_dates_df.index.to_timestamp()
    y_test = np.array(y_test)

    n_plots = len(evaluation_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    plotted = 0

    for result in evaluation_results:
        horizon = result.get("horizon")
        preds = np.array(result.get("predictions"))

        if len(preds) != len(test_index):
            warnings.warn(f"Horizon {horizon}: Prediction length mismatch. Skipping.")
            continue

        r2 = result.get("r2", None)
        title = f"Forecast Horizon: {horizon} months"
        if r2 is not None:
            title += f" (R² = {r2:.2f})"

        ax = axes[plotted]
        ax.plot(test_index, y_test, label="Actual", color="tab:gray", linestyle="--")
        ax.plot(test_index, preds, label="Prediction", color="tab:blue", linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        plotted += 1

    # Hide unused subplots
    for j in range(plotted, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
