"""
Module: model
Defines the MLP neural network architecture used for regression tasks.
"""

from torch import nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers and activations.

    Args:
        in_dim (int): Number of input features.
        layer_dims (list[int]): Sizes of each hidden layer.
        activations (list[str]): Activation functions for each hidden layer.
            Supported values: 'relu', 'tanh', 'sigmoid', 'softplus'.

    Example:
        >>> model = MLP(in_dim=16, layer_dims=[32, 16], activations=['relu', 'tanh'])
    """
    def __init__(self, in_dim: int, layer_dims: list[int], activations: list[str]):
        super().__init__()
        layers = []
        dims = [in_dim] + layer_dims

        # Build each hidden layer: Linear -> Activation
        for i, h in enumerate(layer_dims):
            # Linear transform from dims[i] to dims[i+1]
            layers.append(nn.Linear(dims[i], dims[i+1]))

            act = activations[i].lower()
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            elif act == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif act == 'softplus':
                layers.append(nn.Softplus())
            else:
                raise ValueError(f"Unknown activation '{activations[i]}'")

        # Final layer maps to single output
        layers.append(nn.Linear(dims[-1], 1))

        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, 1).
        """
        return self.net(x)
