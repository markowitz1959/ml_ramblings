import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import tqdm

from model import CTransformer
from data import get_loaders

NUM_CLASSES = 2


def train(args):
    tbw = SummaryWriter(log_dir=args.tb_dir)

    train_loader, test_loader, word2idx = get_loaders(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    print(f'- training batches:   {len(train_loader)}')
    print(f'- validation batches: {len(test_loader)}')

    model = CTransformer(
        emb=args.embedding,
        heads=args.heads,
        depth=args.depth,
        seq_length=args.max_length,
        num_tokens=args.vocab_size,
        num_classes=NUM_CLASSES,
        max_pool=args.max_pool,
        dropout=args.dropout,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(step / (args.lr_warmup / args.batch_size), 1.0)
    )

    seen = 0
    for epoch in range(args.num_epochs):

        print(f'\nepoch {epoch}')
        model.train()

        for ids, labels in tqdm.tqdm(train_loader):
            if torch.cuda.is_available():
                ids, labels = ids.cuda(), labels.cuda()

            if ids.size(1) > args.max_length:
                ids = ids[:, :args.max_length]

            opt.zero_grad()
            out = model(ids)
            loss = F.nll_loss(out, labels)
            loss.backward()

            if args.gradient_clipping > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

            opt.step()
            sch.step()

            seen += ids.size(0)
            tbw.add_scalar('train/loss', loss.item(), seen)

        # validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for ids, labels in test_loader:
                if torch.cuda.is_available():
                    ids, labels = ids.cuda(), labels.cuda()
                if ids.size(1) > args.max_length:
                    ids = ids[:, :args.max_length]
                preds = model(ids).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            acc = correct / total
            print(f'validation accuracy: {acc:.3f}')
            tbw.add_scalar('val/accuracy', acc, epoch)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--num-epochs',         dest='num_epochs',         default=80,      type=int)
    parser.add_argument('--batch-size',         dest='batch_size',         default=4,       type=int)
    parser.add_argument('--lr',                 dest='lr',                 default=1e-4,    type=float)
    parser.add_argument('--embedding',          dest='embedding',          default=128,     type=int)
    parser.add_argument('--vocab-size',         dest='vocab_size',         default=50_000,  type=int)
    parser.add_argument('--max-length',         dest='max_length',         default=512,     type=int)
    parser.add_argument('--heads',              dest='heads',              default=8,       type=int)
    parser.add_argument('--depth',              dest='depth',              default=6,       type=int)
    parser.add_argument('--dropout',            dest='dropout',            default=0.0,     type=float)
    parser.add_argument('--max-pool',           dest='max_pool',           action='store_true')
    parser.add_argument('--lr-warmup',          dest='lr_warmup',          default=10_000,  type=int)
    parser.add_argument('--gradient-clipping',  dest='gradient_clipping',  default=1.0,     type=float)
    parser.add_argument('--tb-dir',             dest='tb_dir',             default='./runs')
    parser.add_argument('--cache-dir',          dest='cache_dir',          default=None)

    args = parser.parse_args()
    print('options:', args)
    train(args)
