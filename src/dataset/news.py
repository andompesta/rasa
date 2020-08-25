from torch.utils.data import Dataset, DataLoader
from os import path
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataset.vocab import Vocab

"""
loader of the 20News dataset
"""
class NewsDataset(Dataset):

    def __init__(self, docs, n_vocab, device):
        """docs are sparse BoW representation."""
        super(NewsDataset, self).__init__()
        self.docs = docs
        self.n_vocab = n_vocab
        self.device = device

    def __getitem__(self, item):
        """Return float32 Bow representation."""
        d = self.docs[item]

        v = np.zeros(self.n_vocab, dtype=np.float32)
        for w, f in d:
            v[w] += f
        return torch.Tensor(v).to(self.device)

    def __len__(self):
        return len(self.docs)

def read_docs(file):
    docs = []
    print('reading', file)
    with open(file) as f:
        for line in f:
            doc = [t.split(':') for t in line.split()[1:]]
            doc = [(int(t)-1, float(f)) for t, f in doc if float(f) > 0]
            if len(doc) > 0:
                docs.append(doc)
            else:
                print('null doc!')
    return docs


def read_dataset(data_dir):
    """Read prefix+train.feat prefix+test.feat prefix+vocab."""
    train_docs = read_docs(path.join(data_dir, 'train.feat'))
    test_docs = read_docs(path.join(data_dir, 'test.feat'))

    print('creating dictionary')
    id2word = []
    with open(path.join(data_dir, 'vocab')) as f:
        id2word.extend([line.strip().split(' ')[0] for line in f])
    dictionary = Vocab(id2word)

    return train_docs, test_docs, dictionary

def load_news_data(data_dir, batch_size, device, dev_ratio=0.):
    train_docs, test_docs, vocab = read_dataset(data_dir)

    if dev_ratio > 0:
        print('splitting train, dev datasets')
        train_docs, dev_docs = train_test_split(train_docs, test_size=dev_ratio, shuffle=True)
        print('train, dev, test', len(train_docs), len(dev_docs), len(test_docs))

        train_loader = DataLoader(NewsDataset(train_docs, len(vocab), device), batch_size, drop_last=False,
                                  num_workers=0)
        dev_loader = DataLoader(NewsDataset(dev_docs, len(vocab), device), batch_size, drop_last=False,
                                num_workers=0)
        test_loader = DataLoader(NewsDataset(test_docs, len(vocab), device), batch_size, drop_last=False,
                                 num_workers=0)

        return train_loader, dev_loader, test_loader, vocab

    else:
        print('train, test', len(train_docs), len(test_docs))

        train_loader = DataLoader(NewsDataset(train_docs, len(vocab), device), batch_size, drop_last=False,
                                  num_workers=0)
        test_loader = DataLoader(NewsDataset(test_docs, len(vocab), device), batch_size, drop_last=False,
                                 num_workers=0)

        return train_loader, test_loader, test_loader, vocab

