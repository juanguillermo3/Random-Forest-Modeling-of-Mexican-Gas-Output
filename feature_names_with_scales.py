"""
title: Feature Names with Scales
description: Utilities for effectively working with units of measurement or scale information when included
             as part of feature names.
"""

#
# (0)
#
import re
import numpy as np
#
# (1)
#
def wipe_out_scale_info(var_name):
    """Removes the measurement scale from the main variable name."""
    return re.sub(r"\s*\(.*?\)", "", var_name).strip()
#
# (2)
#
def clean_variable_name(var_name, max_length=20):
    """
    Cleans variable names for plotting.
    - Removes underscores
    - Converts to lowercase and capitalizes only the first letter
    - Truncates if longer than max_length (adding '...')
    """
    clean_name = var_name.replace("_", " ").lower().capitalize()
    return clean_name[:max_length] + "..." if len(clean_name) > max_length else clean_name
#
# (3)
#
def extract_scale(var_name):
    """Extracts the measurement scale from a variable name (if inside parentheses)."""
    match = re.search(r"\((.*?)\)", var_name)
    return match.group(1) if match else None
