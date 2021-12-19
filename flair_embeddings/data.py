from IPython.display import clear_output

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import json
import os


class Vocab:
    def __init__(self, capacity=10000, save_path='./../vocab'):
        self.capacity = capacity
        self.save_path = save_path

        self.max_len = 0
        self.max_chars = 0

        self.i2c = ['<SEP>', '<BEGIN>', '<END>', '<UNK>']
        self.c2i = {'<SEP>': 0, '<BEGIN>': 1, '<END>': 2, '<UNK>': 3}

        self.i2w = ['<SEP>', '<UNK>']
        self.w2i = {'<SEP>': 0, '<UNK>': 1}

        self.i2t = ['<SEP>', '<UNK>']
        self.t2i = {'<SEP>': 0, '<UNK>': 1}

    def load(self, vocab_path, max_len=200, max_chars=1000):
        with open(os.path.join(vocab_path, 'words.txt'), 'r') as v:
            self.w2i = json.load(v)
            self.i2w = [_ for _ in self.w2i]
            for k in self.w2i:
                self.i2w[self.w2i[k]] = k
        with open(os.path.join(vocab_path, 'chars.txt'), 'r') as v:
            self.c2i = json.load(v)
            self.i2c = [_ for _ in self.c2i]
            for k in self.c2i:
                self.i2c[self.c2i[k]] = k
        with open(os.path.join(vocab_path, 'labels.txt'), 'r') as v:
            self.t2i = json.load(v)
            self.i2t = [_ for _ in self.t2i]
            for k in self.t2i:
                self.i2t[self.t2i[k]] = k

        self.max_len = max_len
        self.max_chars = max_chars

    def save(self, vocab_path):
        with open(os.path.join(vocab_path, 'words.txt'), 'w') as v:
            json.dump(self.w2i, v)
        with open(os.path.join(vocab_path, 'chars.txt'), 'w') as v:
            json.dump(self.c2i, v)
        with open(os.path.join(vocab_path, 'labels.txt'), 'w') as v:
            json.dump(self.t2i, v)

    def read(self, docs):
        for _ in range(self.capacity):
            clear_output()
            print('Reading: ')
            print(_, '/', self.capacity)
            doc = next(docs)
            for sent in doc.sents:
                self.max_len = max(self.max_len, len(sent.tokens))
                chars_len = 0
                for token in sent.morph.tokens:
                    if token.pos not in self.t2i:
                        self.i2t.append(token.pos)
                        self.t2i[token.pos] = len(self.i2t) - 1
                    if token.text not in self.w2i:
                        self.i2w.append(token.text)
                        self.w2i[token.text] = len(self.i2w) - 1
                    for char in token.text:
                        chars_len += 1
                        if char not in self.c2i:
                            self.i2c.append(char)
                            self.c2i[char] = len(self.i2c) - 1
                self.max_chars = max(self.max_chars, chars_len + 10)


class POSDataset(Dataset):
    def __init__(self, vocab, capacity=2000):
        super().__init__()
        self.tokens = []
        self.chars = []
        self.targets = []
        self.capacity = capacity

        self.vocab = vocab

    def __getitem__(self, i):
        return torch.LongTensor(self.chars[i]), torch.LongTensor(self.tokens[i]), torch.LongTensor(self.targets[i])

    def __len__(self):
        return len(self.chars)

    def collate_fn(self, batch):
        tokens, words, pos_tags = list(zip(*batch))
        tokens = pad_sequence(tokens, batch_first=True)
        words = pad_sequence(words, batch_first=True)
        pos_tags = pad_sequence(pos_tags, batch_first=True)
        return tokens, words, pos_tags

    def read(self, docs):
        while len(self.chars) < self.capacity:
            clear_output()
            print('Reading: ')
            print(len(self.chars), '/', self.capacity)
            doc = next(docs)
            for sent in doc.sents:
                self.tokens.append([])
                self.chars.append([])
                self.targets.append([])
                for token in sent.morph.tokens:
                    if token.pos not in self.vocab.t2i:
                        self.targets[-1].append(self.vocab.t2i['<UNK>'])
                    else:
                        self.targets[-1].append(self.vocab.t2i[token.pos])
                    if token.text not in self.vocab.w2i:
                        self.tokens[-1].append(self.vocab.w2i['<UNK>'])
                    else:
                        self.tokens[-1].append(self.vocab.w2i[token.text])
                    self.chars[-1].append(self.vocab.c2i['<BEGIN>'])
                    for char in token.text:
                        if char not in self.vocab.c2i:
                            self.chars[-1].append(self.vocab.c2i['<UNK>'])
                        else:
                            self.chars[-1].append(self.vocab.c2i[char])
                self.chars[-1].append(self.vocab.c2i['<BEGIN>'])