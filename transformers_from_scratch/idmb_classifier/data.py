from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

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

        # pad to max_length
        ids += [PAD] * (self.max_length - len(ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_loaders(vocab_size=50_000, max_length=512, batch_size=4, cache_dir=None):
    raw = load_dataset('imdb', cache_dir=cache_dir)

    word2idx = build_vocab(raw['train']['text'], vocab_size)

    train_ds = IMDBDataset(raw['train'], word2idx, max_length)
    test_ds  = IMDBDataset(raw['test'],  word2idx, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, word2idx
