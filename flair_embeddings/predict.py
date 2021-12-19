import torch

from data import Vocab
from model import FlairEmbeddings, PosTagger

import argparse

import re


class TextProcessor:
    def __init__(self, vocab_path):
        self.vocab = Vocab()
        self.vocab.load(vocab_path)
        self.chars = []
        self.words = []

    def process(self, sent):
        tokens = re.findall(r"[\w']+|[.,!?;]", sent)
        for token in tokens:
            if token not in self.vocab.w2i:
                self.words.append(self.vocab.w2i['<UNK>'])
            else:
                self.words.append(self.vocab.w2i[token])
            self.chars.append(self.vocab.c2i['<BEGIN>'])
            for char in token:
                if char not in self.vocab.c2i:
                    self.chars.append(self.vocab.c2i['<UNK>'])
                else:
                    self.chars.append(self.vocab.c2i[char])
        self.chars.append(self.vocab.c2i['<BEGIN>'])
        return torch.LongTensor(self.chars).unsqueeze(0), torch.LongTensor(self.words).unsqueeze(0)


def predict(args):
    processor = TextProcessor(args['vocab'])
    chars, words = processor.process(args['data'])
    flair_embedding_model = FlairEmbeddings(len(processor.vocab.c2i), processor.vocab.max_len)
    if not args['emb']:
        pos_model = PosTagger(len(processor.vocab.t2i), flair_embedding_model)
    else:
        pos_model = PosTagger(len(processor.vocab.t2i), flair_embedding_model, num_embeddings=len(processor.vocab.w2i))
    pos_model.load_state_dict(torch.load(args['pos_checkpoint_path'], map_location=torch.device('cpu')))
    pos_model.eval()
    return ' '.join([processor.vocab.i2t[pred] for pred in pos_model(chars, words).argmax(dim=-1)[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict POS tags')
    parser.add_argument('--pos_checkpoint_path', action='store',
                        default='./pos.ckpt',
                        help='POS tagger checkpoint')
    parser.add_argument('--data', action='store',
                        default='Просто какое-то предложение',
                        help='Sentence to predict POS tags for')
    parser.add_argument('--vocab', action='store',
                        default='./vocab',
                        help='Vocabulary path')
    parser.add_argument('--emb', action='store_const',
                        const=True, default=False,
                        help='To use classic embeddings')

    args = vars(parser.parse_args())
    print(predict(args))