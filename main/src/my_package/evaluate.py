"""
Module for evaluating a trained MLP model and visualizing its performance.

Provides:
    evaluate_and_plot_mlp(): Load model & norms, compute R² on test data,
    display an error histogram and a true‐vs‐predicted scatter plot.
"""

import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import categories_map, all_inputs, numeric_inputs
from .preprocessing import encode_categoricals
from .model import MLP


def evaluate_and_plot_mlp(
    model_path: str,
    norms_json: str,
    hidden_dims: list[int],
    activations: list[str],
    test_csv: str,
    sample_index: int = None
):
    """
    Load a trained MLP, apply saved normalization constants,
    evaluate on test data, compute R², and plot:

      1) Error histogram (prediction error distribution)
      2) True vs. Predicted scatter (optional highlight of a single sample)

    Args:
        model_path:    Path to the saved model checkpoint (.pth).
        norms_json:    Path to the JSON file with y/feature normalization.
        hidden_dims:   List of hidden layer sizes (must match the trained model).
        activations:   List of activation names for each hidden layer.
        test_csv:      Path to the CSV file with test data.
        sample_index:  If provided, highlights that sample on the scatter plot.
    """

    # Load normalization constants
    with open(norms_json, 'r') as f:
        norms = json.load(f)
    y_mean, y_std = norms['y_mean'], norms['y_std']
    feat_mean, feat_std = norms['feat_mean'], norms['feat_std']

    # Read test data and drop unused columns
    df = pd.read_csv(test_csv)
    for col in ("F2", "F3"):
        df.drop(columns=[col], errors='ignore', inplace=True)

    y_true = df["Output"].to_numpy(dtype=float)

    # Build feature matrix and apply encoding & normalization
    X = df[all_inputs].copy()
    X = encode_categoricals(X)
    for col in numeric_inputs:
        X[col] = (X[col] - feat_mean[col]) / feat_std[col]
    X_np = X.to_numpy(dtype=np.float32)

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(X_np.shape[1], hidden_dims, activations).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Make predictions (normalized)
    with torch.no_grad():
        preds_norm = model(torch.from_numpy(X_np).to(device)).cpu().numpy().flatten()
    y_pred = preds_norm * y_std + y_mean

    # Compute R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² (coefficient of determination): {r2:.4f}")

    # Plot error histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred - y_true, bins=30, edgecolor='k')
    plt.title("Prediction Error Histogram")
    plt.xlabel("Predicted − True")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Plot true vs. predicted values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label='All samples')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', label='45° line')

    # Highlight a specific sample if requested
    if sample_index is not None and 0 <= sample_index < len(y_true):
        plt.scatter(
            [y_true[sample_index]],
            [y_pred[sample_index]],
            color='red', s=80,
            label=f"Sample {sample_index}"
        )

    plt.title("True vs. Predicted")
    plt.xlabel("True Output")
    plt.ylabel("Predicted Output")
    plt.legend()
    plt.tight_layout()
    plt.show()
