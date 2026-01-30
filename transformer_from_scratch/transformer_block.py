# STEP 1 — Position-wise Feed Forward Network

import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# STEP 2 — LayerNorm + Residual Wrapper

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# STEP 3 — GPT-style Decoder Block

from attention_mechanism.attention import MultiHeadAttention   # import your previous code

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.attn(x, x, x, mask))
        x = self.residual2(x, self.ff)
        return x

# STEP 4 — Test Transformer Block
if __name__ == "__main__":
    x = torch.rand(2, 10, 512)

    block = TransformerBlock(d_model=512, num_heads=8)
    out = block(x)

    print("Output shape:", out.shape)

