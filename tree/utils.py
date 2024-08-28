"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=True)
    

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Assuming that if a series has only integer values, it is discrete; otherwise, it is real
    return pd.api.types.is_float_dtype(y) or pd.api.types.is_numeric_dtype(y)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value_counts = Y.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-9))  # Adding epsilon to avoid log(0)



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_counts = Y.value_counts(normalize=True)
    return 1 - np.sum(value_counts ** 2)



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == 'entropy':
        base_entropy = entropy(Y)
        weighted_entropy = 0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_entropy += (len(subset_Y) / len(Y)) * entropy(subset_Y)
        return base_entropy - weighted_entropy

    elif criterion == 'gini':
        base_gini = gini_index(Y)
        weighted_gini = 0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_gini += (len(subset_Y) / len(Y)) * gini_index(subset_Y)
        return base_gini - weighted_gini

    elif criterion == 'MSE':
        base_mse = np.mean((Y - Y.mean()) ** 2)
        weighted_mse = 0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_mse += (len(subset_Y) / len(Y)) * np.mean((subset_Y - subset_Y.mean()) ** 2)
        return base_mse - weighted_mse

    else:
        raise ValueError(f"Unknown criterion: {criterion}")



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -float('inf')
    best_feature = None

    for feature in features:
        gain = information_gain(y, X[feature], criterion)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    """
    if pd.api.types.is_numeric_dtype(X[attribute]):
        mask = X[attribute] <= value
        return X[mask], y[mask], X[~mask], y[~mask]
    else:
        mask = X[attribute] == value
        return X[mask], y[mask], X[~mask], y[~mask]

