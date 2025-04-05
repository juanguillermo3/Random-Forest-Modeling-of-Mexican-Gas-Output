#
def replace_values_with_coordinates(df):
    """
    Replaces all values in the DataFrame with a string of the format {col_name}_{index_value}.

    Parameters:
    - df: DataFrame to tag.

    Returns:
    - A new DataFrame with tagged string values.
    """
    return df.apply(lambda col: [f"{col.name}_{i}" for i in df.index], axis=0)
