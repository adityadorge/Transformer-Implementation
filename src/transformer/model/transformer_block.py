import numpy as np

class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value without mean centering.
    Used in LLaMA, Mistral, and most modern LLMs.
    """

    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: Feature dimension to normalize over
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.weight = np.ones(dim)  # Learnable scale parameter

    def __call__(self, x):
            """
            Apply RMSNorm to input.

            Args:
                x: Input tensor of shape (..., dim)

            Returns:
                Normalized tensor of the same shape
            """
            # Compute RMS along last dimension
            rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
            # Normalize and scale
            return self.weight * (x / rms)   

class MultiHeadAttention:
    """
    Multi-head self-attention mechanism.

    Splits the representation into multiple heads, computes attention
    independently for each head, then concatenates and projects.
    """

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

                   