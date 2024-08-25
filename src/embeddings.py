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


class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initialize the positional encoding.

        Parameters:
        d_model (int): The dimensionality of the model.
        seq_len (int): The maximum length of the input sequence.
        dropout (float): Dropout rate applied after adding positional encoding.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of size (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)

        # Position indices (0, 1, 2, ..., seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Divisor term for positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch size and register as a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Parameters:
        x (torch.Tensor): Input tensor of embeddings with shape (batch_size, seq_len, d_model).

        Returns:
        torch.Tensor: Tensor with positional encoding added and dropout applied.
        """
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
