import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FlairEmbeddings(nn.Module):
    def __init__(self, n_tokens, max_words, embedding_dim=128, hidden_dim=256,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_tokens)

        self.max_words = max_words
        self.n_tokens = n_tokens

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def forward(self, x, hidden=None):
        mask = (x != 0).to(torch.long)

        lengths = mask.sum(dim=1).to('cpu')
        x = self.embedding(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        out = self.dropout(output)
        out = self.fc(out)
        return output, (out, hidden)

    def predict(self, x, hidden=None):
        out, _ = self.forward(x.to(dtype=torch.long), hidden)
        emb = torch.zeros((x.shape[0], self.max_words, self.hidden_dim * 2)).to(device=x.device)
        one_ids = torch.nonzero(x == 1)
        j = 0
        emb[one_ids[0, 0], 0, self.hidden_dim:] = out[one_ids[0, 0], one_ids[0, 1], self.hidden_dim:]
        for i in range(1, one_ids.shape[0]):
            if one_ids[i, 0] != one_ids[i - 1, 0]:
                emb[one_ids[i, 0] - 1, j, self.hidden_dim:] = 0
                j = 0
                emb[one_ids[i, 0], j, self.hidden_dim:] = out[one_ids[i, 0], one_ids[i, 1], self.hidden_dim:]
            else:
                emb[one_ids[i, 0], j, :self.hidden_dim] = out[one_ids[i, 0], one_ids[i, 1], :self.hidden_dim]
                j += 1
                emb[one_ids[i, 0], j, self.hidden_dim:] = out[one_ids[i, 0], one_ids[i, 1], self.hidden_dim:]
        return emb


class PosTagger(nn.Module):
    def __init__(self, output_dim: int, flair, hidden_dim=300,
                 feedforward_dim=100, dropout_rate=0.1,
                 freeze_emb=True, num_embeddings=None, classic_emb_dim=128):
        super(PosTagger, self).__init__()
        self.hidden_dim = hidden_dim
        embedding_dim = flair.hidden_dim * 2
        self.embedder = flair.cpu()
        self.classic_embeddings = None
        if num_embeddings:
            self.classic_embeddings = nn.Embedding(num_embeddings, classic_emb_dim)
        if freeze_emb:
            self.embedder = self.embedder.eval()
            for param in self.embedder.parameters():
                param.requires_grad = False
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True)
        self.feedforward = nn.Linear(2 * hidden_dim, feedforward_dim)
        self.out = nn.Linear(feedforward_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.LongTensor, words: torch.LongTensor):
        mask = (words != 0).to(torch.long)
        lengths = mask.sum(dim=1).to('cpu')

        if self.classic_embeddings:
            x = torch.cat([self.embedder.predict(x), self.classic_embeddings(x)], dim=-1)
        else:
            x = self.embedder.predict(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.encoder(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.feedforward(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.out(x)
