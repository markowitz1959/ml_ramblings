import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        assert emb % heads == 0
        self.emb, self.heads, self.mask = emb, heads, mask

        self.tokeys     = nn.Linear(emb, emb, bias=False)
        self.toqueries  = nn.Linear(emb, emb, bias=False)
        self.tovalues   = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h, s = self.heads, self.emb // self.heads

        keys    = self.tokeys(x).view(b, t, h, s)
        queries = self.toqueries(x).view(b, t, h, s)
        values  = self.tovalues(x).view(b, t, h, s)

        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, s)

        dot = torch.bmm(queries, keys.transpose(1, 2)) / (s ** 0.5)

        if self.mask:  # causal mask: token i cannot attend to positions j > i
            indices = torch.triu_indices(t, t, offset=1, device=x.device)
            dot[:, indices[0], indices[1]] = float('-inf')

        dot = F.softmax(dot, dim=2)
        self.last_attn = dot.detach()  # saved for inspection

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask=False, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.ReLU(),
            nn.Linear(4 * emb, emb),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(self.attention(x) + x)
        x = self.dropout(x)
        x = self.norm2(self.ff(x) + x)
        x = self.dropout(x)
        return x


class CTransformer(nn.Module):
    """Transformer sequence classifier (Bloem 2019)."""

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0):
        super().__init__()
        self.max_pool = max_pool

        self.token_emb = nn.Embedding(num_tokens, emb)
        self.pos_emb   = nn.Embedding(seq_length, emb)

        self.tblocks = nn.Sequential(
            *[TransformerBlock(emb, heads, dropout=dropout) for _ in range(depth)]
        )
        self.toprobs = nn.Linear(emb, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, e = tokens.size()

        positions = torch.arange(t, device=x.device)
        x = self.dropout(tokens + self.pos_emb(positions)[None, :, :].expand(b, t, e))

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        return F.log_softmax(self.toprobs(x), dim=1)


class GTransformer(nn.Module):
    """Transformer for generating text (Bloem 2019). Uses a causal mask so each
    token can only attend to previous tokens."""

    def __init__(self, emb, heads, depth, seq_length, num_tokens, dropout=0.0):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, emb)
        self.pos_emb   = nn.Embedding(seq_length, emb)

        self.tblocks = nn.Sequential(
            *[TransformerBlock(emb, heads, mask=True, dropout=dropout) for _ in range(depth)]
        )
        self.toprobs = nn.Linear(emb, num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, e = tokens.size()

        positions = torch.arange(t, device=x.device)
        x = self.dropout(tokens + self.pos_emb(positions)[None, :, :].expand(b, t, e))
        x = self.tblocks(x)

        # project every position to a distribution over the vocabulary
        x = self.toprobs(x.view(b * t, e)).view(b, t, -1)
        return F.log_softmax(x, dim=2)
