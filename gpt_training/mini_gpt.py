import torch
import torch.nn as nn
import math

from transformer_from_scratch.transformer_block import TransformerBlock  

# STEP 1 — GPT Configuration
class GPTConfig:
    vocab_size = 50257
    block_size = 128
    n_layers = 6
    n_heads = 8
    n_embd = 512
    dropout = 0.1

# STEP 2 — Token + Positional Embeddings

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.n_embd,
                num_heads=config.n_heads,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size


# STEP 3 — Forward Pass + Causal Mask

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

# STEP 4 — Text Generation Function

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

# STEP 5 — Test GPT Forward + Generation

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    x = torch.randint(0, config.vocab_size, (2, 16))
    out = model(x)

    print("Logits shape:", out.shape)

    y = model.generate(x, max_new_tokens=20)
    print("Generated shape:", y.shape)

