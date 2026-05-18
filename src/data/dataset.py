import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(file_path):
    print(f"Loading data from {file_path}...")

    df = pd.read_csv(file_path)

    X = df.drop('num', axis=1)
    y = df['num'].values.ravel()

    # Binary classification
    y = (y > 0).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )


    return X_train, X_test, y_train, y_test

