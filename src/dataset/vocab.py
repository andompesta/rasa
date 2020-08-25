import copy

class Vocab(object):
    def __init__(self, itos):
        super(Vocab, self).__init__()
        self.itos = copy.copy(itos)
        self.stoi = {v: k for k, v in enumerate(itos)}

    def __call__(self, s: str) -> int:
        return self.stoi[s]

    def get_word(self, idx: int) -> str:
        return self.itos[idx]

    def __len__(self):
        return len(self.itos)