import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim=128,
        n_layers=2,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        drop_prob=0.2,
    ):
        super(CharCNNBiLSTM, self).__init__()
        # ponytail: default hidden_dim/n_layers allow zero-arg instantiation; override via kwargs if needed.
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.out_dim = hidden_dim * 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs_1d = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.bilstm = nn.LSTM(
            num_filters * len(kernel_sizes),
            hidden_dim,
            n_layers,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.out_dim, output_size)

    def extract_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs_1d]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1).unsqueeze(1)
        x, _ = self.bilstm(x)
        return self.dropout(x[:, -1, :])

    def forward(self, x):
        return self.fc(self.extract_features(x))


if __name__ == "__main__":
    net = CharCNNBiLSTM(
        vocab_size=96,
        output_size=2,
        embedding_dim=128,
        hidden_dim=64,
        n_layers=2,
        num_filters=100,
    )
    dummy_input = torch.randint(0, 96, (4, 32))
    features = net.extract_features(dummy_input)
    out = net(dummy_input)
    assert out.shape == (4, 2), f"Expected shape (4, 2), got {out.shape}"
    assert features.shape == (4, net.out_dim)
