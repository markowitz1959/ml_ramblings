import gzip
import io
import os
import urllib.request
import zipfile
import torch
import numpy as np

NUM_TOKENS = 256  # byte-level vocabulary — all possible byte values 0-255

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
    """
    Load enwik8 from a local file and split into train / val / test.
    Download handled automatically by maybe_download().
    """
    with gzip.open(path) if path.endswith('.gz') else open(path, 'rb') as f:
        data = np.frombuffer(f.read(n_train + n_valid + n_test), dtype=np.uint8)

    train, val, test = np.split(data, [n_train, n_train + n_valid])
    return torch.from_numpy(train.copy()), torch.from_numpy(val.copy()), torch.from_numpy(test.copy())


def sample_batch(data, seq_length, batch_size):
    """
    Slice out a random batch of subsequences from the data.
    Input:  data[start : start + seq_length]
    Target: data[start+1 : start + seq_length + 1]  (shifted one step ahead)
    """
    starts = torch.randint(low=0, high=data.size(0) - seq_length - 1, size=(batch_size,))
    inputs  = torch.stack([data[s:s + seq_length]     for s in starts]).long()
    targets = torch.stack([data[s + 1:s + seq_length + 1] for s in starts]).long()
    return inputs, targets
