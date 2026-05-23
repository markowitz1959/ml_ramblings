
# This is based on the transformer from scratch article available here:
# https://peterbloem.nl/blog/transformers


import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads, self.mask = k, heads, mask

        # These compute the queries, keys and values for all
        # heads
        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)


    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        s = k // h

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # Fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Scaled dot-product attention
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / (s ** 0.5)

        if self.mask:
            indices = torch.triu_indices(t, t, offset=1, device=x.device)
            dot[:, indices[0], indices[1]] = float("-inf")

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)

        # Unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        self.tblocks = nn.Sequential(
            *[TransformerBlock(k=k, heads=heads) for _ in range(depth)]
        )

        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        positions = torch.arange(t, device=x.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
