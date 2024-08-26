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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initialize the FeedForward module.

        Parameters:
        d_model (int): The dimensionality of the input and output.
        d_ff (int): The dimensionality of the intermediate layer.
        dropout (float): The dropout rate to apply after the first linear transformation.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First fully connected layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Second fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForward network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        torch.Tensor: Output tensor of the same shape as input.
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after the first linear layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply the second linear layer
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        """
        Initialize the MultiHeadAttention module.

        Parameters:
        d_model (int): Dimensionality of the model (input and output dimensions).
        heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply after the attention scores.
        """
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads  # Dimensionality of each head

        # Define linear layers for query, key, and value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        """
        Compute the scaled dot-product attention.

        Parameters:
        query (torch.Tensor): Query tensor of shape (batch_size, heads, seq_len, d_k).
        key (torch.Tensor): Key tensor of shape (batch_size, heads, seq_len, d_k).
        value (torch.Tensor): Value tensor of shape (batch_size, heads, seq_len, d_k).
        mask (torch.Tensor): Optional mask tensor to apply (shape should be broadcastable).
        dropout (nn.Dropout): Optional dropout layer to apply after softmax.

        Returns:
        torch.Tensor: The output of the attention mechanism.
        torch.Tensor: The attention scores (useful for analysis or visualization).
        """
        d_k = query.shape[-1]
        # Scaled dot-product attention
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply the mask (if any)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_scores = attention_scores.softmax(dim=-1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # Multiply attention weights by values
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None) -> torch.Tensor:
        """
        Forward pass through the MultiHeadAttention module.

        Parameters:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        mask (torch.Tensor): Optional mask tensor.

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Project input vectors to query, key, and value vectors
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and transpose for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        # Compute attention and apply to values
        x, self.attention_scores = self.attention(query, key, value, mask)

        # Reshape the output back to (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # Apply the final linear transformation
        return self.w_o(x)
