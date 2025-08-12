"""
Module: training
Implements k‐fold cross‐validation training for the MLP, collecting loss
and R² histories, and returns trained models and metrics.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from .model import MLP
from .preprocessing import prepare_data


def _select_loss(loss_type: str):
    """
    Return the appropriate PyTorch loss given a string identifier.

    Args:
        loss_type: 'mse' for mean squared error, 'huber' for Smooth L1 loss.

    Returns:
        A torch.nn loss module.

    Raises:
        ValueError if an unsupported loss_type is provided.
    """
    if loss_type == 'mse':
        return torch.nn.MSELoss()
    elif loss_type == 'huber':
        return torch.nn.SmoothL1Loss()
    else:
        raise ValueError("loss_type must be 'mse' or 'huber'")


def cross_validate_mlp_with_history(
    df,
    k_folds: int,
    layer_dims: list[int],
    activations: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    random_state: int,
    loss_type: str = 'mse'
):
    """
    Perform k‐fold cross‐validation on an MLP regressor, recording training
    and validation loss & R² at each epoch, and return the models and metrics.

    Args:
        df:             pandas DataFrame containing 'Output' and feature columns.
        k_folds:        Number of folds for KFold CV.
        layer_dims:     Hidden layer sizes for the MLP.
        activations:    Activation names for each hidden layer.
        epochs:         Number of training epochs per fold.
        lr:             Learning rate for Adam optimizer.
        batch_size:     Batch size for DataLoader.
        device:         'cpu' or 'cuda'.
        random_state:   Seed for reproducible splitting.
        loss_type:      'mse' or 'huber' (Smooth L1).

    Returns:
        fold_results:   List of dicts with keys:
                        - 'fold'
                        - 'train_true', 'train_pred'
                        - 'val_true',   'val_pred'
        histories:      List of tuples (train_losses, val_losses, train_r2s, val_r2s).
        models:         List of trained MLP instances (one per fold).
        norms:          Normalization constants (y_mean, y_std, feat_mean, feat_std).
        feature_names:  List of feature column names after preprocessing.
    """
    # Prepare data: feature matrix, normalized target, norms, and feature names
    X_np, y, norms, feature_names = prepare_data(df)
    # Get the chosen loss function
    loss_fn = _select_loss(loss_type)

    # Set up KFold splitter
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_results, histories, models = [], [], []

    # Iterate over each train/validation split
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_np), start=1):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        print(f"Fold {fold} validation indices: {va_idx.tolist()}")

        # Create PyTorch datasets & loaders
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        ld_va = DataLoader(ds_va, batch_size=batch_size)

        # Initialize model and optimizer
        model = MLP(X_tr.shape[1], layer_dims, activations).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Lists to store per‐epoch metrics
        tr_losses, va_losses, tr_r2s, va_r2s = [], [], [], []

        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in ld_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

            # Evaluate on full train & val sets
            model.eval()
            with torch.no_grad():
                p_tr = model(torch.from_numpy(X_tr).to(device)).cpu().numpy().flatten()
                p_va = model(torch.from_numpy(X_va).to(device)).cpu().numpy().flatten()

            # Compute loss according to loss_type
            if loss_type == 'mse':
                tr_l = mean_squared_error(y_tr.flatten(), p_tr)
                va_l = mean_squared_error(y_va.flatten(), p_va)
            else:
                # For Huber, use the same loss_fn on numpy arrays
                tr_l = loss_fn(torch.from_numpy(p_tr).unsqueeze(1),
                               torch.from_numpy(y_tr)).item()
                va_l = loss_fn(torch.from_numpy(p_va).unsqueeze(1),
                               torch.from_numpy(y_va)).item()

            # Record losses and R² scores
            tr_losses.append(tr_l)
            va_losses.append(va_l)
            tr_r2s.append(r2_score(y_tr.flatten(), p_tr))
            va_r2s.append(r2_score(y_va.flatten(), p_va))

        # Detailed per‐sample validation errors (pred − true)
        val_errors = (p_va - y_va.flatten()).tolist()
        print(f"Validation errors (pred-true): {val_errors}")
        print(f"Final losses (train, val): ({tr_losses[-1]:.4f}, {va_losses[-1]:.4f})\n")

        # Store fold outputs
        fold_results.append({
            'fold':       fold,
            'train_true': y_tr.flatten(),
            'train_pred': p_tr,
            'val_true':   y_va.flatten(),
            'val_pred':   p_va,
        })
        histories.append((tr_losses, va_losses, tr_r2s, va_r2s))
        models.append(model)

    # Return results plus normalization info and feature list
    return fold_results, histories, models, norms, feature_names
