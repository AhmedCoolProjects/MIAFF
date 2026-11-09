import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        drop_prob=0.2,
        **kwargs,
    ):
        super().__init__()
        # ponytail: kwargs absorbs unused architecture params; drop when caller configs stabilize.
        self.num_filters = num_filters
        self.out_dim = len(kernel_sizes) * num_filters
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs_1d = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.out_dim, output_size)

    def extract_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [
            F.max_pool1d(F.relu(conv(x)), kernel_size=x.size(2)).squeeze(2)
            for conv in self.convs_1d
        ]
        return self.dropout(torch.cat(x, dim=1))

    def forward(self, x):
        return self.fc(self.extract_features(x))


if __name__ == "__main__":
    net = CharCNN(vocab_size=96, output_size=2, embedding_dim=128)
    dummy_input = torch.randint(0, 96, (4, 64))
    features = net.extract_features(dummy_input)
    out = net(dummy_input)
    assert out.shape == (4, 2), f"Expected shape (4, 2), got {out.shape}"
    assert features.shape == (4, net.out_dim)

    print("Test passed")
