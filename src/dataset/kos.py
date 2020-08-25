from torch.utils.data import Dataset, DataLoader
from os import path
import torch
import numpy as np
from src.dataset.vocab import Vocab

class KosDataset(Dataset):

    def __init__(self, docs, n_vocab, device):
        """docs are sparse BoW representation."""
        super(KosDataset, self).__init__()
        self.docs = docs
        self.n_vocab = n_vocab
        self.device = device

    def __getitem__(self, item):
        """Return float32 Bow representation."""
        d = self.docs[item]
        return torch.Tensor(d).to(self.device)

    def __len__(self):
        return len(self.docs)

def load_kos_data(args, device):
    # read data
    train = np.load(path.join(args.data_dir, "mat_A.npy"))
    test = np.load(path.join(args.data_dir, "mat_B.npy"))

    def format(data: np.array):
        start = data[:, 0].min()
        D = data[:, 0].max() + 1
        W = data[:, 1].max() + 1
        docs = []
        for d in np.arange(start, D):
            doc = np.zeros(W)
            doc[data[data[:, 0] == d, 1]] = data[data[:, 0] == d, 2]
            assert (doc > 0).sum() == data[data[:, 0] == d].shape[0]
            assert doc.sum() > 0
            docs.append(doc)
        return docs

    train = format(train)
    test = format(test)

    # vocab
    id2word = dict()
    for id, w in enumerate(np.load(path.join(args.data_dir, "words.npy"))):

        id2word[id] = w

    vocab = Vocab(id2word)

    train_loader = DataLoader(KosDataset(train, len(vocab), device), args.batch_size,
                              drop_last=False,
                              num_workers=0)
    test_loader = DataLoader(KosDataset(test, len(vocab), device), args.batch_size,
                             drop_last=False,
                             num_workers=0)

    return train_loader, test_loader, test_loader, vocab


if __name__ == '__main__':
    from main import parse_args
    args = parse_args()
    train, dev, test, vocab = load_kos_data(args, "cpu")


    for docs_t in dev:
        print(docs_t)
        break