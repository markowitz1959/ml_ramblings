import gzip
import os
import urllib.request
import zipfile
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

# ─── Device ───────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'using device: {device}')

# ──────────────────────────────────────────────────────────────────────────────

# ─── Data ─────────────────────────────────────────────────────────────────────

NUM_TOKENS  = 256
ENWIK8_URL  = 'http://mattmahoney.net/dc/enwik8.zip'
ENWIK8_PATH = os.path.expanduser('~/data/enwik8')


def maybe_download(path=ENWIK8_PATH, url=ENWIK8_URL):
    if os.path.exists(path):
        print(f'enwik8 already present at {path}')
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    zip_path = path + '.zip'
    print(f'downloading enwik8 to {zip_path} ...')
    urllib.request.urlretrieve(url, zip_path)
    print('extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract('enwik8', os.path.dirname(path))
    os.remove(zip_path)
    print('done.')
    return path


def load_enwik8(path=ENWIK8_PATH, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    with gzip.open(path) if path.endswith('.gz') else open(path, 'rb') as f:
        data = np.frombuffer(f.read(n_train + n_valid + n_test), dtype=np.uint8)
    train, val, test = np.split(data, [n_train, n_train + n_valid])
    return torch.from_numpy(train.copy()), torch.from_numpy(val.copy()), torch.from_numpy(test.copy())


def sample_batch(data, seq_length, batch_size):
    starts  = torch.randint(low=0, high=data.size(0) - seq_length - 1, size=(batch_size,))
    inputs  = torch.stack([data[s:s + seq_length]         for s in starts]).long()
    targets = torch.stack([data[s + 1:s + seq_length + 1] for s in starts]).long()
    return inputs, targets


path = maybe_download()
data_train, data_val, data_test = load_enwik8(path)
print(f'- train: {data_train.size(0):,} bytes')
print(f'- val:   {data_val.size(0):,} bytes')
print(f'- test:  {data_test.size(0):,} bytes')

# ──────────────────────────────────────────────────────────────────────────────

# ─── Model ────────────────────────────────────────────────────────────────────

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

        if self.mask:
            indices = torch.triu_indices(t, t, offset=1, device=x.device)
            dot[:, indices[0], indices[1]] = float('-inf')

        dot = F.softmax(dot, dim=2)
        self.last_attn = dot.detach()

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


class GTransformer(nn.Module):
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
        x = self.toprobs(x.view(b * t, e)).view(b, t, -1)
        return F.log_softmax(x, dim=2)

# ──────────────────────────────────────────────────────────────────────────────

# ─── Text generation and evaluation ──────────────────────────────────────────

def bits_per_byte(model, data, seq_length, batch_size=32, num_batches=50):
    """Estimate bits per byte on a data split using random batches."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            source, target = sample_batch(data, seq_length, batch_size)
            source, target = source.to(device), target.to(device)
            output = model(source)
            loss = F.nll_loss(output.transpose(2, 1), target)
            total_loss += loss.item()
    # convert from nats to bits (nll_loss returns natural log)
    return (total_loss / num_batches) / torch.log(torch.tensor(2.0)).item()


def generate(model, seed, seq_length, length, temperature):
    model.eval()
    sequence = seed.clone()
    with torch.no_grad():
        for _ in range(length):
            context  = sequence[-seq_length:].unsqueeze(0).to(device)
            logprobs = model(context)[0, -1, :]
            if temperature == 0.0:
                next_byte = logprobs.argmax()
            else:
                probs     = F.softmax(logprobs / temperature, dim=0)
                next_byte = torch.multinomial(probs, 1).squeeze()
            sequence = torch.cat([sequence, next_byte.unsqueeze(0).cpu()])
    return ''.join(chr(b) if 32 <= b < 127 else '.' for b in sequence.tolist())

# ──────────────────────────────────────────────────────────────────────────────

# ─── Parameters ───────────────────────────────────────────────────────────────

NUM_BATCHES       = 100_000  # total number of training steps
BATCH_SIZE        = 32       # number of random windows per step
SEQ_LENGTH        = 256      # context window — how many bytes the model can see back
EMBEDDING         = 128      # size of each token embedding vector
HEADS             = 8        # number of attention heads
DEPTH             = 4        # number of transformer blocks
DROPOUT           = 0.0      # dropout rate (0 = off)
LR                = 1e-4     # learning rate
LR_WARMUP         = 5_000    # number of steps to ramp lr up from 0 to LR
GRADIENT_CLIPPING = 1.0      # max gradient norm (0 = off)
PRINT_EVERY       = 500      # print loss every N steps
SAMPLE_EVERY      = 1_500    # generate a text sample every N steps
SAMPLE_LENGTH     = 300      # number of bytes to generate in each sample
TEMPERATURE       = 0.5      # sampling temperature (lower = more conservative)
TB_DIR            = './runs'

# ──────────────────────────────────────────────────────────────────────────────

# ─── Training ─────────────────────────────────────────────────────────────────

model = GTransformer(
    emb=EMBEDDING,
    heads=HEADS,
    depth=DEPTH,
    seq_length=SEQ_LENGTH,
    num_tokens=NUM_TOKENS,
    dropout=DROPOUT,
).to(device)

print(f'- parameters: {sum(p.numel() for p in model.parameters()):,}')

opt = torch.optim.Adam(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda step: min(step / (LR_WARMUP / BATCH_SIZE), 1.0)
)

tbw = SummaryWriter(log_dir=TB_DIR)

print('\nstarting training...')
model.train()

for step in tqdm.trange(NUM_BATCHES):

    opt.zero_grad()

    source, target = sample_batch(data_train, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    source, target = source.to(device), target.to(device)

    output = model(source)
    loss   = F.nll_loss(output.transpose(2, 1), target)
    loss.backward()

    if GRADIENT_CLIPPING > 0:
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

    opt.step()
    sch.step()

    tbw.add_scalar('train/loss', loss.item(), step)

    if step > 0 and step % PRINT_EVERY == 0:
        train_bpb = bits_per_byte(model, data_train, SEQ_LENGTH)
        val_bpb   = bits_per_byte(model, data_val,   SEQ_LENGTH)
        gap       = val_bpb - train_bpb
        print(f'\nstep {step:,}  loss: {loss.item():.4f}  |  train bpb: {train_bpb:.4f}  val bpb: {val_bpb:.4f}  gap: {gap:.4f}')
        tbw.add_scalar('bpb/train', train_bpb, step)
        tbw.add_scalar('bpb/val',   val_bpb,   step)
        tbw.add_scalar('bpb/gap',   gap,        step)
        print('  attention breakdown per layer:')
        for i, block in enumerate(model.tblocks):
            dot    = block.attention.last_attn
            self_  = dot.diagonal(dim1=1, dim2=2).mean().item()
            left1_ = dot.diagonal(offset=-1, dim1=1, dim2=2).mean().item()
            left2_ = dot.diagonal(offset=-2, dim1=1, dim2=2).mean().item()
            other_ = 1.0 - self_ - left1_ - left2_
            print(f'  layer {i}: self={self_:.3f}  prev1={left1_:.3f}  prev2={left2_:.3f}  other={other_:.3f}')
        model.train()

    if step > 0 and step % SAMPLE_EVERY == 0:

        seed_start = random.randint(0, data_val.size(0) - SEQ_LENGTH)
        seed = data_val[seed_start:seed_start + SEQ_LENGTH].long()
        print(f'\n--- generated text at step {step:,} ---')
        print(generate(model, seed, SEQ_LENGTH, SAMPLE_LENGTH, TEMPERATURE))
        print('---')
        model.train()




# ─── Final evaluation ────────────────────────────────────────────────────────

print('\n=== final bits per byte ===')
train_bpb = bits_per_byte(model, data_train, SEQ_LENGTH, num_batches=200)
val_bpb   = bits_per_byte(model, data_val,   SEQ_LENGTH, num_batches=200)
test_bpb  = bits_per_byte(model, data_test,  SEQ_LENGTH, num_batches=200)
print(f'train: {train_bpb:.4f}  |  val: {val_bpb:.4f}  |  test: {test_bpb:.4f}')
print(f'(random baseline: 8.0 bpb)')

# ─── Post-training inference ──────────────────────────────────────────────────

# generate from a few random positions in the validation set
print('\n\n=== post-training samples ===')
for i in range(3):
    seed_start = random.randint(0, data_val.size(0) - SEQ_LENGTH)
    seed = data_val[seed_start:seed_start + SEQ_LENGTH].long()
    seed_text = ''.join(chr(b) if 32 <= b < 127 else '.' for b in seed.tolist())
    print(f'\n--- sample {i+1} ---')
    print(f'[seed]: {seed_text[-80:]}')  # show last 80 chars of seed as context
    print(f'[generated]:')
    print(generate(model, seed, SEQ_LENGTH, SAMPLE_LENGTH, TEMPERATURE))

# generate from custom text prompts
prompts = [
    'The history of artificial intelligence',
    'In mathematics, a prime number is',
    'The French Revolution began in',
    'Albert Einstein was born in',
    'The Python programming language',
    'The United States of America',
    'Charles Darwin proposed the theory of',
    'The solar system consists of',
    'William Shakespeare wrote',
    'The Roman Empire fell in',
    'DNA is a molecule that',
    'The first world war started in',
    'Leonardo da Vinci was',
    'In economics, inflation refers to',
    'The Amazon rainforest is located in',
]

for prompt in prompts:
    seed = torch.tensor([ord(c) for c in prompt], dtype=torch.long)
    print(f'\n--- prompt: "{prompt}" ---')
    print(generate(model, seed, SEQ_LENGTH, SAMPLE_LENGTH, TEMPERATURE))
