from torch import nn


class TrackModel(nn.Module):
    def __init__(self):
        super().__init__()

        input_count = 16

        self.net = nn.Sequential(
            nn.Linear(input_count, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
