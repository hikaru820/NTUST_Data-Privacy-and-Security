# DataConversion_Kanon.py
# Reads Anonymized Dataset
import numpy as np

def parse_range(val):
    val = str(val).strip()
    if val == '*':   return np.nan
    if '-' in val:
        lo, hi = val.split('-')
        return (float(lo) + float(hi)) / 2
    return float(val)

def load_anonymized(k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(f'../adult_k{k}.csv')

    # Same drops as original
    df = df.drop(['fnlwgt', 'education', 'native-country'], axis=1)

    # Parse numeric QI range strings to midpoints
    for col in ['age', 'educational-num', 'hours-per-week']:
        df[col] = df[col].apply(parse_range)
        df[col] = df[col].fillna(df[col].mean())

    y = df['income']
    x = df.drop('income', axis=1)

    x = pd.get_dummies(x, columns=['workclass', 'marital-status', 'occupation',
                                    'relationship', 'race', 'gender'])

    scaler = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    x[num_cols] = scaler.fit_transform(x[num_cols])

    return train_test_split(x, y, test_size=0.2, random_state=42)