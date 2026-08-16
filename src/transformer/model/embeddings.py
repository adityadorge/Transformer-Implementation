import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
        Sinusoidal Positional Encoding implemented from scratch.
        
        Args:
            d_model (int): Hidden dimension size of the model.
            max_len (int): Maximum sequence length to pre-compute.
            dropout (float): Dropout probability applied to the combined embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. Instantiate a zero matrix for positions up to max_len
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)

        # 2. Create position indices vector: shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 3. Calculate division term in log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # 4. Fill even indices (0, 2, 4...) with sin, odd indices (1, 3, 5...) with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. Add a batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 6. Register as persistent buffer (non-trainable state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token embeddings of shape (Batch_Size, Seq_Len, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encodings added, shape (Batch_Size, Seq_Len, d_model)
        """
        seq_len = x.size(1)
        
        # Element-wise addition of token embeddings and pre-computed positional encodings
        x = x + self.pe[:, :seq_len]
        
        return self.dropout(x)
        

class EmbeddingWithProjection(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_embed: int, 
        d_model: int, 
        max_position_embeddings: int = 512, 
        dropout: float = 0.1,
        padding_idx: int = None,
        use_bias: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        self.padding_idx = padding_idx

        # 1. Initialize Low-Dimensional Embedding Matrix W_embed: shape (V, d_embed)
        embed_data = torch.randn(vocab_size, d_embed)
        if padding_idx is not None:
            embed_data[padding_idx] = 0.0
        self.W_embed = nn.Parameter(embed_data)

        # 2. Initialize Projection Weight W_proj: shape (d_embed, d_model)
        # Drawn from Normal Distribution N(0, std^2) where std = 1 / sqrt(d_embed)
        std = 1.0 / math.sqrt(d_embed)
        proj_data = torch.rand(d_embed, d_model) * std
        self.W_proj = nn.Parameter(proj_data)

        if use_bias:
            bias_data = torch.zeros(d_model)
            self.b_proj = nn.Parameter(bias_data)
        else:
            self.register_parameter('b_proj', None)

        # Scaling factor & Positional Encoding Module
        self.scaling = float(math.sqrt(self.d_model))
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, max_len=max_position_embeddings)

        # Post-processing
        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.long, f'Input tensor must have dtype torch.long, got {x.dtype}' 

        # Bounds check
        if (x < 0).any() or (x >= self.vocab_size).any():
            raise IndexError(f"Indices must be in range [0, {self.vocab_size - 1}]")

        # Zero out padding vector if padding_idx is set
        if self.padding_idx is not None:
            with torch.no_grad():
                self.W_embed[self.padding_idx].fill_(0.0)  

        # 1. Token embedding lookup and matrix projection from scratch
        embedded = self.W_embed[x]
        projected = torch.matmul(embedded, self.W_proj)    

        if self.b_proj is not None:
            projection = projected + self.b_proj

        token_embedding = projection * self.scaling

        # 2. Add positional encoding via the separate class
        embeddings = self.pos_encoding(token_embedding)

        # 3. Apply normalization and dropout
        normalized_sum = self.layernorm(embeddings)
        final_output = self.dropout(normalized_sum)

        return final_output  


if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    vocab_size = 1000
    d_embed = 128
    d_model = 512

    # Dummy Token Input (Batch of Token IDs)
    input_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    print(f"Input Token IDs Shape: {input_token_ids.shape}")
    print(input_token_ids)

    # Instantiate Module
    custom_emb_module = EmbeddingWithProjection(
        vocab_size=vocab_size,
        d_embed=d_embed,
        d_model=d_model,
        max_position_embeddings=512,
        dropout=0.1
    )

    # Forward Pass
    output_embeddings = custom_emb_module(input_token_ids)
    print(f"Output Embeddings Shape: {output_embeddings.shape}")

    # Verify Backward Pass & Gradient Flow back to Custom Parameter Table
    loss = output_embeddings.sum()
    loss.backward()

    print(f"\nGradient Verification:")
    print(f"Embedding Weight Matrix Shape: {custom_emb_module.W_embed.shape}")
    print(f"Embedding Grad Has Values:    {custom_emb_module.W_embed.grad is not None}")
    print(f"Non-Zero Gradient Rows Count:  {(custom_emb_module.W_embed.grad.sum(dim=-1) != 0).sum().item()}")



        
        
