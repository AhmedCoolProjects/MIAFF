import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim=128,
        n_layers=2,
        drop_prob=0.2,
    ):
        super(CharLSTM, self).__init__()
        # ponytail: default hidden_dim/n_layers allow zero-arg instantiation; override via kwargs if needed.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.out_dim, output_size)

    def extract_features(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.dropout(lstm_out[:, -1, :])

    def forward(self, x):
        return self.fc(self.extract_features(x))

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
        )
        return hidden


if __name__ == "__main__":
    net = CharLSTM(vocab_size=96, output_size=2, embedding_dim=128, hidden_dim=64, n_layers=2)
    dummy_input = torch.randint(0, 96, (4, 32))
    features = net.extract_features(dummy_input)
    out = net(dummy_input)
    assert out.shape == (4, 2), f"Expected shape (4, 2), got {out.shape}"
    assert features.shape == (4, net.out_dim)

