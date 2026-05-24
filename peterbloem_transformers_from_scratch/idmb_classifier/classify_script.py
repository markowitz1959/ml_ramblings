import sys
sys.path.insert(0, '/Users/pcharmoy/dev/ml_ramblings/peterbloem_transformers_from_scratch/idmb_classifier')

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from model import CTransformer
from data import get_loaders

NUM_CLASSES = 2


def train(params):
    tbw = SummaryWriter(log_dir=params['tb_dir'])

    train_loader, test_loader, word2idx = get_loaders(
        vocab_size=params['vocab_size'],
        max_length=params['max_length'],
        batch_size=params['batch_size'],
        cache_dir=params['cache_dir'],
    )

    print(f'- training batches:   {len(train_loader)}')
    print(f'- validation batches: {len(test_loader)}')

    model = CTransformer(
        emb=params['embedding'],
        heads=params['heads'],
        depth=params['depth'],
        seq_length=params['max_length'],
        num_tokens=params['vocab_size'],
        num_classes=NUM_CLASSES,
        max_pool=params['max_pool'],
        dropout=params['dropout'],
    )

    if torch.cuda.is_available():
        model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(step / (params['lr_warmup'] / params['batch_size']), 1.0)
    )

    seen = 0
    for epoch in range(params['num_epochs']):

        print(f'\nepoch {epoch}')
        model.train()

        train_correct, train_total = 0, 0
        for ids, labels in tqdm.tqdm(train_loader):
            if torch.cuda.is_available():
                ids, labels = ids.cuda(), labels.cuda()

            if ids.size(1) > params['max_length']:
                ids = ids[:, :params['max_length']]

            opt.zero_grad()
            out = model(ids)
            loss = F.nll_loss(out, labels)
            loss.backward()

            if params['gradient_clipping'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clipping'])

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
                if torch.cuda.is_available():
                    ids, labels = ids.cuda(), labels.cuda()
                if ids.size(1) > params['max_length']:
                    ids = ids[:, :params['max_length']]
                preds = model(ids).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            val_acc = correct / total
            print(f'train accuracy: {train_acc:.3f}  |  validation accuracy: {val_acc:.3f}')
            tbw.add_scalar('train/accuracy', train_acc, epoch)
            tbw.add_scalar('val/accuracy',   val_acc,   epoch)


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


if __name__ == '__main__':
    params = dict(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        embedding=EMBEDDING,
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        heads=HEADS,
        depth=DEPTH,
        dropout=DROPOUT,
        max_pool=MAX_POOL,
        lr_warmup=LR_WARMUP,
        gradient_clipping=GRADIENT_CLIPPING,
        tb_dir=TB_DIR,
        cache_dir=CACHE_DIR,
    )
    print('parameters:', params)
    train(params)
