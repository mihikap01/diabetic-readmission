"""Shared preprocessing functions for diabetic readmission data."""

import numpy as np
import pandas as pd
from config import DATA_DIR


def load_data():
    """Load and return the diabetic readmission dataset."""
    df = pd.read_csv(DATA_DIR / "diabetic_data.csv")
    df = df.replace("?", np.nan)
    return df


def fix_age(age):
    """Convert age range strings to midpoint numeric values."""
    age_dict = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    return age_dict.get(age, np.nan)


def fix_diag(diag):
    """Clean diagnostic codes by removing letter prefixes."""
    if isinstance(diag, str):
        if diag[0] in ['V', 'E']:
            return diag[1:]
        if diag == '?':
            return np.nan
    return diag


def fix_weight(weight):
    """Convert weight range strings to midpoint numeric values."""
    weight_dict = {
        '[0-25)': 12, '[25-50)': 37, '[50-75)': 62, '[75-100)': 87,
        '[100-125)': 112, '[125-150)': 137, '[150-175)': 162, '[175-200)': 187
    }
    return weight_dict.get(weight, np.nan)


def preprocess_data(df):
    """Apply all standard preprocessing steps to the dataframe."""
    df = df.copy()
    df['age'] = df['age'].apply(fix_age)
    df['weight'] = df['weight'].apply(fix_weight)
    df['diag_1'] = df['diag_1'].apply(fix_diag)
    df['diag_2'] = df['diag_2'].apply(fix_diag)
    df['diag_3'] = df['diag_3'].apply(fix_diag)
    return df
