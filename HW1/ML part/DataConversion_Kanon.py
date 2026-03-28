# DataConversion_Kanon.py
# Reads Anonymized Dataset
import numpy as np
from pathlib import Path

def parse_range(val):
    val = str(val).strip()
    if val == '*':   return np.nan
    if '-' in val:
        parts = val.split('-')
        lo, hi = parts[0], parts[-1]   # take first and last to be safe
        try:
            return (float(lo) + float(hi)) / 2
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

def load_anonymized(k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(base_dir.parent / f'adult_k{k}.csv')

    # Same drops as original
    df = df.drop(['fnlwgt', 'education', 'native-country'], axis=1)

    if df['income'].dtype == object:
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    y = df['income']
    x = df.drop('income', axis=1)

    # Parse numeric QI range strings to midpoints
    for col in ['age', 'educational-num', 'hours-per-week']:
        x[col] = x[col].apply(parse_range)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    for col in ['age', 'educational-num', 'hours-per-week']:
        mean_value = x_train[col].mean()
        x_train[col] = x_train[col].fillna(mean_value)
        x_test[col] = x_test[col].fillna(mean_value)

    x_train = pd.get_dummies(x_train, columns=['workclass', 'marital-status', 'occupation',
                                               'relationship', 'race', 'gender'])
    x_test = pd.get_dummies(x_test, columns=['workclass', 'marital-status', 'occupation',
                                             'relationship', 'race', 'gender'])

    all_columns = x_train.columns.union(x_test.columns)
    x_train = x_train.reindex(columns=all_columns, fill_value=0)
    x_test = x_test.reindex(columns=all_columns, fill_value=0)

    scaler = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    return x_train, x_test, y_train, y_test
