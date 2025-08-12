"""
Module: sweep
Visualize how the trained MLP’s output changes when sweeping Input 10
across a specified range, overlaying each sample’s true value.
"""

import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil

from .config import categories_map, all_inputs, numeric_inputs
from .preprocessing import encode_categoricals
from .model import MLP


def sweep_plot_samples_mlp(
    model_path: str,
    norms_json: str,
    hidden_dims: list[int],
    activations: list[str],
    test_csv: str,
    sample_index: int = -1
):
    """
    For each test sample, sweep Input 10 over [0,3000], plot the MLP’s
    predicted output curve, and overlay the true (Input10, Output) point.

    Args:
        model_path:    Path to saved model checkpoint (.pth).
        norms_json:    Path to JSON file with normalization constants.
        hidden_dims:   Hidden layer sizes (must match trained model).
        activations:   Activation names for each hidden layer.
        test_csv:      CSV file containing test features + 'Output' column.
        sample_index:  
            >=0 : show only that sample;
            -1  : grid of all samples (3 plots per row);
            -2  : one figure per sample.
    """

    # Load normalization constants
    with open(norms_json, 'r') as f:
        norms = json.load(f)
    y_mean, y_std = norms['y_mean'], norms['y_std']
    feat_mean, feat_std = norms['feat_mean'], norms['feat_std']

    # Read and preprocess test data
    df = pd.read_csv(test_csv)
    # Drop unused columns if present
    for col in ('F2', 'F3'):
        df.drop(columns=[col], errors='ignore', inplace=True)

    # Extract true targets
    y_true = df['Output'].to_numpy(dtype=float)

    # Build feature matrix and apply encoding/normalization
    X = df[all_inputs].copy()
    X = encode_categoricals(X)
    for col in numeric_inputs:
        X[col] = (X[col] - feat_mean[col]) / feat_std[col]
    Z_np = X.to_numpy(dtype=np.float32)

    # Store Input 10 normalization for sweeping
    mu10, sig10 = feat_mean['Input 10'], feat_std['Input 10']

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(Z_np.shape[1], hidden_dims, activations).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Helper to plot one sample’s sweep
    def plot_sample(i, ax):
        # Base feature vector for sample i
        Zi = Z_np[i:i+1]
        xi, yi = df['Input 10'].iloc[i], y_true[i]

        # Generate a grid of raw Input 10 values
        grid = np.linspace(0, 3000, 300)
        grid_norm = (grid - mu10) / sig10

        # Repeat sample’s features, replace Input 10 with each grid point
        Zr = np.repeat(Zi, len(grid), axis=0)
        idx10 = all_inputs.index('Input 10')
        Zr[:, idx10] = grid_norm

        # Predict (normalized), then denormalize
        with torch.no_grad():
            p_norm = model(torch.from_numpy(Zr).to(device)).cpu().numpy().flatten()
        p = p_norm * y_std + y_mean

        # Plot curve + true sample point
        ax.plot(grid, p, linewidth=1)
        ax.scatter(xi, yi, color='red', s=20, zorder=5)
        ax.set_xlim(0, 3000)
        ax.set_xticks([0, 3000])
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("Input 10")
        ax.set_ylabel("Output")

    N = len(y_true)

    # Single‐sample view
    if 0 <= sample_index < N:
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_sample(sample_index, ax)
        plt.tight_layout()
        plt.show()

    # One figure per sample
    elif sample_index == -2:
        for i in range(N):
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_sample(i, ax)
            plt.tight_layout()
            plt.show()

    # Grid view of all samples
    else:
        cols = 3
        rows = ceil(N / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6))
        axs = axs.flatten()
        for i in range(N):
            plot_sample(i, axs[i])
        # Remove any extra axes
        for ax in axs[N:]:
            fig.delaxes(ax)
        plt.tight_layout()
        plt.show()
