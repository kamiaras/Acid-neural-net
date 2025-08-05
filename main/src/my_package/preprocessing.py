"""
Module: preprocessing
Contains functions to clean data, encode categorical features,
and normalize targets and numeric features for model input.
"""

import pandas as pd
import numpy as np
from .config import categories_map, all_inputs, numeric_inputs


def drop_unused(df: pd.DataFrame, to_drop=('F2', 'F3')) -> pd.DataFrame:
    """
    Remove columns that are not needed for modeling.

    Args:
        df:       Input DataFrame.
        to_drop:  Tuple of column names to drop if present.

    Returns:
        A DataFrame with the specified columns removed.
    """
    return df.drop(columns=[c for c in to_drop if c in df.columns])


def normalize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Standardize the target variable to zero mean and unit variance.

    Args:
        y:  Array of shape (n_samples, 1).

    Returns:
        y_norm:   Normalized target array.
        y_mean:   Mean of original y.
        y_std:    Standard deviation of original y (or 1 if zero).
    """
    μ = y.mean()
    σ = y.std() if y.std() != 0 else 1.0
    return (y - μ) / σ, float(μ), float(σ)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features based on the predefined categories_map.

    - Binary features (2 categories) are mapped to ±1.
    - Multi-category features are one-hot encoded, ensuring all categories appear.

    Args:
        X: DataFrame containing categorical columns.

    Returns:
        DataFrame with encoded categorical features.
    """
    for col, cats in categories_map.items():
        if col not in X:
            continue

        if len(cats) == 2:
            # Binary encode as +1 / -1
            X[col] = X[col].map({cats[0]: +1.0, cats[1]: -1.0})
        else:
            # One-hot encode multi-category feature
            X[col] = pd.Categorical(X[col], categories=cats)
            dummies = pd.get_dummies(X[col], prefix=col)

            # Ensure every category has a column, even if absent in data
            for c in cats:
                name = f"{col}_{c}"
                if name not in dummies:
                    dummies[name] = 0

            # Replace original column with its dummies
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    return X


def normalize_numeric(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardize numeric features to zero mean and unit variance.

    Args:
        X: DataFrame containing numeric_inputs columns.

    Returns:
        X_norm:   DataFrame with normalized numeric columns.
        mean:     Series of means for each numeric feature.
        std:      Series of std deviations (zeros replaced by 1.0).
    """
    mean = X[numeric_inputs].mean()
    std = X[numeric_inputs].std().replace(0, 1.0)
    X[numeric_inputs] = (X[numeric_inputs] - mean) / std
    return X, mean, std


def prepare_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline: drop unused, normalize target,
    encode & normalize features.

    Args:
        df: Raw DataFrame with at least 'Output' and all_inputs columns.

    Returns:
        X_np:            NumPy array of processed features.
        y:               Normalized target array of shape (n_samples, 1).
        norms:           Dict with y_mean, y_std, feat_mean, feat_std.
        feature_names:   List of feature column names after encoding.
    """
    # 1) Drop unused columns
    df_clean = drop_unused(df)

    # 2) Normalize target
    y_array = df_clean['Output'].to_numpy(dtype=np.float32).reshape(-1, 1)
    y, y_mean, y_std = normalize_target(y_array)

    # 3) Prepare features
    X = df_clean[all_inputs].copy()
    X = encode_categoricals(X)
    X, feat_mean, feat_std = normalize_numeric(X)

    # 4) Convert to NumPy
    feature_names = X.columns.to_list()
    X_np = X.to_numpy(dtype=np.float32)

    # 5) Collect normalization constants
    norms = {
        'y_mean':   y_mean,
        'y_std':    y_std,
        'feat_mean': feat_mean.to_dict(),
        'feat_std':  feat_std.to_dict()
    }

    return X_np, y, norms, feature_names
