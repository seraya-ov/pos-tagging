from data import Vocab, POSDataset
from trainer import POSTrainer, FlairTrainer
from model import FlairEmbeddings, PosTagger

from torch.utils.data import DataLoader

import argparse
from nerus import load_nerus


def train(args):
    docs = load_nerus(args['data'])
    vocab = Vocab(capacity=args['vocab_size'])
    if args['vocab'] != '':
        vocab.load(args['vocab'])
    else:
        vocab.read(docs)
    docs = load_nerus(args['data'])
    train_dataset = POSDataset(vocab, capacity=args['train_size'])
    train_dataset.read(docs)
    test_dataset = POSDataset(vocab, capacity=args['val_size'])
    test_dataset.read(docs)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], collate_fn=train_dataset.collate_fn)
    flair_embedding_model = FlairEmbeddings(len(vocab.c2i), vocab.max_len)
    flair_trainer = FlairTrainer(flair_embedding_model, train_loader, test_loader, save_every=5,
                                 save_path=args['checkpoint_path'], lr=3e-4)
    flair_trainer.fit(20, args['cuda'], log=args['log'])
    pos_model = PosTagger(len(vocab.t2i), flair_embedding_model)
    pos_trainer = POSTrainer(pos_model, train_loader, test_loader, save_every=1,
                             save_path=args['checkpoint_path'], lr=args['lr'])
    pos_trainer.fit(args['epochs'], args['cuda'], log=args['log'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the POS tagging model')
    parser.add_argument('--checkpoint_path', action='store',
                        default='./checkpoints/',
                        help='Checkpoint path')
    parser.add_argument('--train_size', action='store',
                        default=100000,
                        help='Train set size')
    parser.add_argument('--val_size', action='store',
                        default=20000,
                        help='Val set size')
    parser.add_argument('--batch_size', action='store',
                        default=128,
                        help='Batch size')
    parser.add_argument('--epochs', action='store',
                        default=10,
                        help='Epochs')
    parser.add_argument('--vocab', action='store',
                        default='',
                        help='Vocabulary path')
    parser.add_argument('--lr',  action='store',
                        default=0.003,
                        help='Learning rate')
    parser.add_argument('--log', action='store_const',
                        const=True, default=False,
                        help='Log metrics to wandb')
    parser.add_argument('--cuda', action='store_const',
                        const=True, default=False,
                        help='Train on CUDA')
    parser.add_argument('--data', action='store',
                        default='./nerus_lenta.conllu.gz',
                        help='path to a data archive file (collnu format (default: https://github.com/natasha/nerus))')
    args = vars(parser.parse_args())
    train(args)
