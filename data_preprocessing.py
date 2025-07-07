import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(clean_csv_path):
    # --- load cleaned data ---
    df = pd.read_csv(clean_csv_path)
    
    # --- explicit column lists ---
    cat_cols     = [f"Input {i}" for i in range(1, 8)]
    num_cols     = [f"Input {i}" for i in range(8, 18)] + ["input_18"]
    output_col   = "Output 1"
    quality_cols = ["F2", "F3"]
    
    # --- fill numeric NaNs with col mean ---
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")  # ensure numeric
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    # --- one-hot encode categoricals, NaN→all-zeros ---
    cat_dfs = []
    for c in cat_cols:
        # cast to string first (so categories like 1.0 vs 1 are consistent)
        s = df[c].astype("category")
        dummies = pd.get_dummies(s, prefix=c, dtype=float)
        missing = df[c].isna()
        if missing.any():
            dummies.loc[missing, :] = 0.0
        cat_dfs.append(dummies)
    df_cat = pd.concat(cat_dfs, axis=1).reset_index(drop=True)
    
    # --- normalize numeric inputs ---
    df_num = df[num_cols].reset_index(drop=True)
    scaler_X = StandardScaler()
    df_num[num_cols] = scaler_X.fit_transform(df_num[num_cols])
    
    # --- normalize output ---
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(df[[output_col]]).ravel().astype(np.float32)
    
    # --- assemble final feature matrix ---
    X = np.hstack([df_cat.values.astype(np.float32),
                   df_num.values.astype(np.float32)])
    
    # --- pull out quality for later sampling ---
    quality = df[quality_cols].astype(np.float32).values
    
    return X, y, quality, scaler_X, scaler_y
