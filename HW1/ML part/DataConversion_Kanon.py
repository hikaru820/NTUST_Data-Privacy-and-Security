# DataConversion_Kanon.py
# Reads Anonymized Dataset
import numpy as np

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

    # Align columns with the original feature space so the model can be evaluated
    # against x_test_orig (which includes native-country and original category values).
    # Generalized-only columns (e.g. race_*, race_Non-White) are dropped because they
    # don't exist in the original; suppressed columns (native-country_*) are filled with 0.
    orig = pd.read_csv('../adult.csv')
    orig = orig.replace(r'^\s*\?\s*$', np.nan, regex=True).dropna()
    orig = orig.drop(['fnlwgt', 'education', 'income'], axis=1)
    orig_dummies = pd.get_dummies(orig, columns=['workclass', 'marital-status', 'occupation',
                                                  'relationship', 'race', 'gender',
                                                  'native-country'])
    x = x.reindex(columns=orig_dummies.columns, fill_value=0)

    scaler = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    x[num_cols] = scaler.fit_transform(x[num_cols])

    return train_test_split(x, y, test_size=0.2, random_state=42)