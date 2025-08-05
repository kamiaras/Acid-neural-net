import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_train_scatter(model, X, y, scaler_y=None, device=None):
    """
    Plots a scatter of true vs. predicted values on the training data.
    
    Args:
        model      : your trained torch.nn.Module
        X          : numpy array (N×D) or torch Tensor of inputs
        y          : numpy array (N,) or torch Tensor of true targets (normalized)
        scaler_y   : (optional) sklearn StandardScaler fitted on the output,
                     so that you can inverse-transform back to the original scale
        device     : (optional) torch.device to move X/model to
    """
    model.eval()
    # Prepare inputs
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X)
    else:
        X_tensor = X
    if device is not None:
        X_tensor = X_tensor.to(device)
        model.to(device)
    # Predict
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().ravel()
    # True values
    if isinstance(y, torch.Tensor):
        true_vals = y.cpu().numpy().ravel()
    else:
        true_vals = y
    
    # Inverse-transform if scaler provided
    if scaler_y is not None:
        preds     = scaler_y.inverse_transform(preds.reshape(-1,1)).ravel()
        true_vals = scaler_y.inverse_transform(true_vals.reshape(-1,1)).ravel()
    
    # Plot
    plt.figure()
    plt.scatter(true_vals, preds, alpha=0.6)
    # Identity line
    mn = min(true_vals.min(), preds.min())
    mx = max(true_vals.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.title('Training Data: True vs. Predicted')
    plt.tight_layout()
    plt.show()
