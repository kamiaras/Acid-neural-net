"""
Module: plotting
Provides functions to visualize training and validation performance of the MLP.
"""

import matplotlib.pyplot as plt

def plot_diagnostics(
    folds: list[dict],
    histories: list[tuple[list, list, list, list]],
    norms: dict,
    k_folds: int,
    epochs: int,
    save_path: str = None,
    show: bool = True
):
    """
    Plot diagnostics for each fold of cross‐validation.

    Generates, for each fold:
      1. Loss vs. Epoch (training & validation)
      2. R² vs. Epoch  (training & validation)
      3. True vs. Predicted scatter (training set)
      4. True vs. Predicted scatter (validation set)

    Args:
        folds:      List of dicts with keys 'fold', 'train_true', 'train_pred',
                    'val_true', and 'val_pred'.
        histories:  List of tuples (train_losses, val_losses, train_r2s, val_r2s).
        norms:      Dict with 'y_mean' and 'y_std' to denormalize predictions.
        k_folds:    Number of folds (rows in the plot grid).
        epochs:     Number of training epochs (x‐axis length).
        save_path:  If provided, path to save the figure (PNG).
        show:       If True, display the plot with `plt.show()`.
    """
    # Create a grid of subplots: k_folds rows x 4 columns
    fig, axes = plt.subplots(k_folds, 4, figsize=(20, 4 * k_folds))
    
    for i, ((fr, (tr_l, va_l, tr_r2, va_r2))) in enumerate(zip(folds, histories)):
        ax1, ax2, ax3, ax4 = axes[i]

        # 1) Loss vs. Epoch
        ax1.plot(range(1, epochs + 1), tr_l, label='train')
        ax1.plot(range(1, epochs + 1), va_l, label='val')
        ax1.set_title(f'Fold {fr["fold"]} Loss')
        ax1.set_ylim(0, 2)
        ax1.legend()

        # 2) R² vs. Epoch
        ax2.plot(range(1, epochs + 1), tr_r2, label='train')
        ax2.plot(range(1, epochs + 1), va_r2, label='val')
        ax2.set_title(f'Fold {fr["fold"]} R²')
        ax2.set_ylim(0, 1)
        ax2.legend()

        # 3) Train True vs Predicted
        yt_train = fr['train_true'] * norms['y_std'] + norms['y_mean']
        yp_train = fr['train_pred'] * norms['y_std'] + norms['y_mean']
        ax3.scatter(yt_train, yp_train, s=10, alpha=0.6)
        lim = [min(yt_train.min(), yp_train.min()), max(yt_train.max(), yp_train.max())]
        ax3.plot(lim, lim, 'k--')
        ax3.set_title(f'Fold {fr["fold"]} Train')

        # 4) Validation True vs Predicted
        yt_val = fr['val_true'] * norms['y_std'] + norms['y_mean']
        yp_val = fr['val_pred'] * norms['y_std'] + norms['y_mean']
        ax4.scatter(yt_val, yp_val, s=10, alpha=0.6)
        lim = [min(yt_val.min(), yp_val.min()), max(yt_val.max(), yp_val.max())]
        ax4.plot(lim, lim, 'k--')
        ax4.set_title(f'Fold {fr["fold"]} Val')

    plt.tight_layout()

    # Save to file if requested
    if save_path:
        fig.savefig(save_path)

    # Display interactively if requested
    if show:
        plt.show()
