import torch
import math
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the input embedding layer.

        Parameters:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the model.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Define an embedding layer that maps vocabulary indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Parameters:
        x (torch.Tensor): Input tensor of token indices.

        Returns:
        torch.Tensor: Scaled embedding tensor.
        """
        # Scale the embeddings by the square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)
