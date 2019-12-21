from torch import nn


class TrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 9, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 21, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(21),
            nn.AvgPool2d(kernel_size=2)
        )

        cnn_out_count = 21

        self.net = nn.Sequential(
            nn.Linear(cnn_out_count, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.net(x)
