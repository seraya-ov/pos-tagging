import wandb

import torch
import torch.nn as nn

import os


class FlairTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, lr=2e-5, betas=(0.9, 0.999),
                 project="flair_embeddings", save_every=None, save_path='./'):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = project
        wandb.init(project=project)

    def train_epoch(self, cuda=True, clip=5):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        for batch_idx, (tokens, _, __) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
            x = tokens
            y_f = tokens[:, 1:]
            y_b = tokens[:, :-1]
            output_f, output_b, _ = self.model(x.to(dtype=torch.long))[1]
            loss_forward = self.criterion(output_f.reshape(-1, output_f.shape[-1]).to(dtype=torch.float),
                                          y_f.reshape(-1).to(dtype=torch.long))
            loss_backward = self.criterion(output_b.reshape(-1, output_b.shape[-1]).to(dtype=torch.float),
                                           y_b.reshape(-1).to(dtype=torch.long))
            loss = loss_forward + loss_backward

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()

            print('\rTrain loss: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), batch_idx + 1, len(self.train_loader)), end='')
        print()
        loss = total_loss / len(self.train_loader)
        return loss

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch_idx, (tokens, _, __) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                x = tokens
                y_f = tokens[:, 1:]
                y_b = tokens[:, :-1]
                output_f, output_b, _ = self.model(x.to(dtype=torch.long))[1]
                loss_forward = self.criterion(output_f.reshape(-1, output_f.shape[-1]).to(dtype=torch.float),
                                              y_f.reshape(-1).to(dtype=torch.long))
                loss_backward = self.criterion(output_b.reshape(-1, output_b.shape[-1]).to(dtype=torch.float),
                                               y_b.reshape(-1).to(dtype=torch.long))
                loss = loss_forward + loss_backward

                total_loss += loss.item()

                print('\rVal loss: %4f, Batch: %d of %d' % (
                    total_loss / (batch_idx + 1), batch_idx + 1, len(self.val_loader)), end='')
            print()
            loss = total_loss / len(self.val_loader)
            return loss

    @staticmethod
    def log(epoch, train_loss, test_loss):
        wandb.log({
            'train': {
                'loss': train_loss,
            },
            'val': {
                'loss': test_loss,
            },
            'epoch': epoch
        })

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))

    def fit(self, max_epochs: int = 20, cuda=True, clip=5, log=False):
        for epoch in range(max_epochs):
            if epoch and self.save_every and epoch % self.save_every == 0:
                self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            train_loss = self.train_epoch(cuda=cuda, clip=clip)
            test_loss = self.test_epoch(cuda=cuda)
            if log:
                self.log(epoch, train_loss, test_loss)


class POSTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, lr=3e-3, betas=(0.9, 0.999),
                 project="flair_pos_tagger", save_every=None, save_path='./', name=None):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.save_path = save_path
        self.save_every = save_every
        self.name = project
        wandb.init(project=project, name=name)

    def train_epoch(self, cuda=True, clip=5):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (tokens, words, pos_tags) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if cuda:
                tokens = tokens.cuda()
                pos_tags = pos_tags.cuda()
                words = words.cuda()

            output = self.model(tokens.to(dtype=torch.long), words)
            loss = self.criterion(output.view(-1, output.shape[-1]),
                                  pos_tags.view(-1).to(dtype=torch.long))

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()

            mask = (words != 0).to(torch.long)
            pred = torch.argmax(output, dim=-1)
            correct += ((pred == pos_tags)*mask).sum().item()
            total += mask.sum().item()
            print('\rTrain loss: %4f, Train accuracy: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), correct / total, batch_idx + 1, len(self.train_loader)
            ), end='')
        print()
        loss, accuracy = total_loss / len(self.train_loader), correct / total
        return loss, accuracy

    def test_epoch(self, cuda=True):
        if cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            total = 0
            correct = 0
            for batch_idx, (tokens, words, pos_tags) in enumerate(self.val_loader):
                if cuda:
                    tokens = tokens.cuda()
                    pos_tags = pos_tags.cuda()
                    words = words.cuda()

                output = self.model(tokens.to(dtype=torch.long), words)
                loss = self.criterion(output.view(-1, output.shape[-1]),
                                      pos_tags.view(-1).to(dtype=torch.long))
                total_loss += loss.item()

                mask = (words != 0).to(torch.long)
                pred = torch.argmax(output, dim=-1)
                correct += ((pred == pos_tags) * mask).sum().item()
                total += mask.sum().item()

                print('\rVal loss: %4f, Val accuracy: %4f, Batch: %d of %d' % (
                    total_loss / (batch_idx + 1), correct / total, batch_idx + 1, len(self.val_loader)
                ), end='')
            print()
            loss, accuracy = total_loss / len(self.val_loader), correct / total
            return loss, accuracy

    @staticmethod
    def log(epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        wandb.log({
            'train': {
                'loss': train_loss,
                'acc': train_accuracy
            },
            'val': {
                'loss': test_loss,
                'acc': test_accuracy
            },
            'epoch': epoch
        })

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))

    def fit(self, max_epochs: int = 20, cuda=True, clip=5, log=False):
        for epoch in range(max_epochs):
            if epoch and self.save_every and epoch % self.save_every == 0:
                self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            train_loss, train_accuracy = self.train_epoch(cuda=cuda, clip=clip)
            test_loss, test_accuracy = self.test_epoch(cuda=cuda)
            if log:
                self.log(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
        if self.save_every:
            self.checkpoint(max_epochs)