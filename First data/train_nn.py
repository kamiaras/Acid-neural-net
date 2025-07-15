import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def train_nn(X, y,
             hidden_dims,
             activation='ReLU',
             lr=1e-3,
             weight_decay=0.0,
             epochs=50,
             batch_size=32):
    """
    Train a feed-forward NN with optional nonlinearity.

    activation: name of a torch.nn activation (e.g. 'ReLU', 'Tanh'), 
                or 'linear' / 'none' for no activation (fully linear network).
    """
    # normalize activation name
    act_key = activation.strip().lower()

    # build model layers
    layers = []
    in_dim = X.shape[1]
    for h in hidden_dims:
        # linear layer
        layers.append(nn.Linear(in_dim, h))
        # optionally add activation
        if act_key not in ('linear', 'none'):
            act_cls = getattr(nn, activation)
            layers.append(act_cls())
        in_dim = h
    # final output layer (no activation)
    layers.append(nn.Linear(in_dim, 1))

    model = nn.Sequential(*layers)

    # data loader
    dataset = TensorDataset(torch.from_numpy(X).float(),
                            torch.from_numpy(y).unsqueeze(1).float())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()

    # training loop
    loss_history = []
    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}  MSE={epoch_loss:.4f}")

    # plot training loss
    plt.figure()
    plt.plot(np.arange(1, epochs+1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.tight_layout()
    plt.show()

    return model, loss_history
