from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tqdm

NUM_CLASSES = 2

# ─── Device ───────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'using device: {device}')

# ──────────────────────────────────────────────────────────────────────────────

# ─── Model ────────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8):
        super().__init__()
        assert emb % heads == 0
        self.emb, self.heads = emb, heads

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
        dot = F.softmax(dot, dim=2)
        self.last_attn = dot.detach()  # (b*h, t, t) — saved for inspection

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads)
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

# ──────────────────────────────────────────────────────────────────────────────

# ─── Data ─────────────────────────────────────────────────────────────────────

PAD, UNK = 0, 1

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, vocab_size):
    counts = Counter(token for text in texts for token in tokenize(text))
    most_common = [word for word, _ in counts.most_common(vocab_size - 2)]
    word2idx = {word: idx + 2 for idx, word in enumerate(most_common)}
    word2idx['<pad>'] = PAD
    word2idx['<unk>'] = UNK
    return word2idx

class IMDBDataset(Dataset):
    def __init__(self, examples, word2idx, max_length):
        self.examples = examples
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text  = self.examples[idx]['text']
        label = self.examples[idx]['label']
        tokens = tokenize(text)[:self.max_length]
        ids = [self.word2idx.get(t, UNK) for t in tokens]
        ids += [PAD] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def get_loaders(vocab_size, max_length, batch_size, cache_dir):
    raw = load_dataset('imdb', cache_dir=cache_dir)
    word2idx = build_vocab(raw['train']['text'], vocab_size)
    train_ds = IMDBDataset(raw['train'], word2idx, max_length)
    test_ds  = IMDBDataset(raw['test'],  word2idx, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, word2idx

# ──────────────────────────────────────────────────────────────────────────────

# ─── Parameters ───────────────────────────────────────────────────────────────

NUM_EPOCHS        = 80
BATCH_SIZE        = 4
LR                = 1e-4
EMBEDDING         = 128     # size of each token embedding vector
VOCAB_SIZE        = 50_000  # number of words to keep in the vocabulary
MAX_LENGTH        = 512     # maximum sequence length (reviews are truncated to this)
HEADS             = 8       # number of attention heads
DEPTH             = 6       # number of transformer blocks
DROPOUT           = 0.0     # dropout rate (0 = off)
MAX_POOL          = True    # True = max pooling, False = mean pooling
LR_WARMUP         = 10_000  # number of steps to ramp lr up from 0 to LR
GRADIENT_CLIPPING = 1.0     # max gradient norm (0 = off)
TB_DIR            = './runs'
CACHE_DIR         = '~/data'  # where to store the IMDB dataset

# ──────────────────────────────────────────────────────────────────────────────

# ─── Training ─────────────────────────────────────────────────────────────────

tbw = SummaryWriter(log_dir=TB_DIR)

train_loader, test_loader, word2idx = get_loaders(
    vocab_size=VOCAB_SIZE,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    cache_dir=CACHE_DIR,
)

print(f'- training batches:   {len(train_loader)}')
print(f'- validation batches: {len(test_loader)}')

model = CTransformer(
    emb=EMBEDDING,
    heads=HEADS,
    depth=DEPTH,
    seq_length=MAX_LENGTH,
    num_tokens=VOCAB_SIZE,
    num_classes=NUM_CLASSES,
    max_pool=MAX_POOL,
    dropout=DROPOUT,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda step: min(step / (LR_WARMUP / BATCH_SIZE), 1.0)
)

seen = 0
for epoch in range(NUM_EPOCHS):

    print(f'\nepoch {epoch}')
    model.train()

    train_correct, train_total = 0, 0
    for ids, labels in tqdm.tqdm(train_loader):
        ids, labels = ids.to(device), labels.to(device)

        if ids.size(1) > MAX_LENGTH:
            ids = ids[:, :MAX_LENGTH]

        opt.zero_grad()
        out = model(ids)
        loss = F.nll_loss(out, labels)
        loss.backward()

        if GRADIENT_CLIPPING > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        opt.step()
        sch.step()

        train_correct += (out.argmax(dim=1) == labels).sum().item()
        train_total   += labels.size(0)
        seen += ids.size(0)
        tbw.add_scalar('train/loss', loss.item(), seen)

    train_acc = train_correct / train_total

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for ids, labels in test_loader:
            ids, labels = ids.to(device), labels.to(device)
            if ids.size(1) > MAX_LENGTH:
                ids = ids[:, :MAX_LENGTH]
            preds = model(ids).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        val_acc = correct / total
        print(f'train accuracy: {train_acc:.3f}  |  validation accuracy: {val_acc:.3f}')
        tbw.add_scalar('train/accuracy', train_acc, epoch)
        tbw.add_scalar('val/accuracy',   val_acc,   epoch)

    # attention statistics per layer (computed on last training batch of each epoch)
    print('  attention breakdown per layer:')
    for i, block in enumerate(model.tblocks):
        dot = block.attention.last_attn  # (b*h, t, t)
        self_  = dot.diagonal(dim1=1, dim2=2).mean().item()
        left_  = dot.diagonal(offset=-1, dim1=1, dim2=2).mean().item()
        right_ = dot.diagonal(offset=1,  dim1=1, dim2=2).mean().item()
        other_ = 1.0 - self_ - left_ - right_
        print(f'  layer {i}: self={self_:.3f}  left={left_:.3f}  right={right_:.3f}  other={other_:.3f}')

# ─── Inference ────────────────────────────────────────────────────────────────

def predict(text, model, word2idx, max_length=MAX_LENGTH):
    model.eval()
    tokens = tokenize(text)[:max_length]
    ids = [word2idx.get(t, UNK) for t in tokens]
    ids += [PAD] * (max_length - len(ids))
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).exp()
    neg, pos = probs[0]
    label = 'POSITIVE' if pos > neg else 'NEGATIVE'
    print(f'{label}  (pos={pos:.2f}, neg={neg:.2f})')
    print(f'text: {text[:80]}...')

# a few hand-written examples
predict("This film was absolutely brilliant, one of the best I've ever seen.", model, word2idx)
predict("Terrible movie, complete waste of time. The acting was awful.", model, word2idx)
predict("It was okay, nothing special but not bad either.", model, word2idx)

# a few real examples from the test set
raw = load_dataset('imdb', cache_dir=CACHE_DIR)
print()
for i in [0, 1, 2]:
    text  = raw['test'][i]['text']
    label = raw['test'][i]['label']
    print(f'true label: {"POSITIVE" if label == 1 else "NEGATIVE"}')
    predict(text, model, word2idx)
    print()
