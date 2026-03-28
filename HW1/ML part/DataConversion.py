# DataConversion.py
# Reads dataset and converts it to a suitable format for ML

import pandas as pd
import numpy as np
from pathlib import Path

from utils import debug_print, set_debug

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_original_data():
    set_debug(False)

    base_dir = Path(__file__).resolve().parent
    rawDF = pd.read_csv(base_dir.parent / 'adult.csv')
    #debug_print(rawDF.head(10))

    cleanDF = rawDF.replace(r'^\s*\?\s*$', np.nan, regex=True)

    cleanDF = cleanDF.drop('fnlwgt', axis=1)  # Drop the 'fnlwgt' column as it is not relevant for our analysis
    cleanDF = cleanDF.drop('education', axis=1)  # Drop the 'education' column as it is redundant with 'education-num'
    cleanDF = cleanDF.dropna()  # Drop rows with missing values to ensure the dataset is clean for analysis

    # Check how many rows were removed
    print(f"Original shape: {rawDF.shape}")
    print(f"Cleaned shape: {cleanDF.shape}")

    debug_print(cleanDF.head(5))

    # Convert the 'income' column to binary values (0 and 1)
    cleanDF['income'] = cleanDF['income'].map({'<=50K': 0, '>50K': 1}) 
    # One-hot encode the categorical strings to numerical values for machine learning algorithms
    df = pd.get_dummies(cleanDF, columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'])

    # Split the dataset into training and testing sets
    y = df['income']
    x = df.drop('income', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    return x_train, x_test, y_train, y_test