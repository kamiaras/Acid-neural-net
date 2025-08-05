import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype

def preprocess_data(clean_csv_path, use_mask):
    """
    Load a cleaned CSV and preprocess only the selected inputs,
    with fixed categorical classes for Inputs 1–8.

    Args:
      clean_csv_path (str): path to the cleaned CSV with columns
        Input 1…Input 18, Output 1, F2, F3.
      use_mask (list or array of bool, length 18): mask[i]=True means
        keep "Input {i+1}", mask[i]=False means drop it.

    Returns:
      X         : numpy array of shape (N, D_selected)
      y         : numpy array of shape (N,)
      quality   : numpy array of shape (N, 2)
      scaler_X  : StandardScaler fitted on numeric inputs (or None)
      scaler_y  : StandardScaler fitted on the output
    """
    df = pd.read_csv(clean_csv_path)

    # Fixed categories for Inputs 1–8
    categories_map = {
        'Input 1': [1, 2],
        'Input 2': [1, 2, 3, 4, 5],
        'Input 3': [1, 2, 3, 4, 5],
        'Input 4': list(range(1, 18)),   # 1–17
        'Input 5': list(range(1, 6)),    # 1–5
        'Input 6': [1, 2, 3],
        'Input 7': [1, 2, 3, 4],
        'Input 8': list(range(1, 11)),   # 1–10
    }

    # 1) all 18 input names
    input_names = [f"Input {i}" for i in range(1, 19)]
    if len(use_mask) != 18:
        raise ValueError("use_mask must be length 18")

    # 2) select those to keep
    selected = [name for name, keep in zip(input_names, use_mask) if keep]

    # 3) split selected into categorical (1–8) and numeric (9–18)
    cat_cols = []
    num_cols = []
    for name in selected:
        idx = int(name.split()[1])
        if 1 <= idx <= 8:
            cat_cols.append(name)
        else:
            num_cols.append(name)

    output_col   = "Output 1"
    quality_cols = ["F2", "F3"]

    # 4) numeric: coerce & fill
    if num_cols:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # 5) one-hot encode categoricals (using fixed classes, NaN→all-zero)
    cat_dfs = []
    for c in cat_cols:
        dtype = CategoricalDtype(categories=categories_map[c], ordered=False)
        s = df[c].astype(dtype)
        d = pd.get_dummies(s, prefix=c, dtype=float)
        cat_dfs.append(d)
    df_cat = pd.concat(cat_dfs, axis=1).reset_index(drop=True) if cat_dfs else pd.DataFrame(index=df.index)

    # 6) normalize numeric inputs
    if num_cols:
        df_num = df[num_cols].reset_index(drop=True)
        scaler_X = StandardScaler()
        df_num[num_cols] = scaler_X.fit_transform(df_num[num_cols])
    else:
        df_num   = pd.DataFrame(index=df.index)
        scaler_X = None

    # 7) normalize output
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(df[[output_col]]).ravel().astype(np.float32)

    # 8) assemble X
    parts = []
    if not df_cat.empty:
        parts.append(df_cat.values.astype(np.float32))
    if not df_num.empty:
        parts.append(df_num.values.astype(np.float32))
    X = np.hstack(parts) if parts else np.empty((len(df), 0), dtype=np.float32)

    # 9) pull out quality
    quality = df[quality_cols].astype(np.float32).values

    return X, y, quality, scaler_X, scaler_y
