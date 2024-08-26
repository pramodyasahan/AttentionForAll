import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the LayerNormalization module.

        Parameters:
        eps (float): A small constant added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps = eps
        # Learnable parameters for scaling and shifting the normalized output
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor (gamma)
        self.beta = nn.Parameter(torch.zeros(1))  # Shifting factor (beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, ..., d_model).

        Returns:
        torch.Tensor: The normalized tensor with the same shape as the input.
        """
        # Calculate the mean and standard deviation along the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # Normalize the input tensor and apply the learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
