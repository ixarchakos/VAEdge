import torch.nn as nn


class SmallNet(nn.Module):
    def __init__(self, in_channels):

        """
        in_channels:  int, number of input channels
        out_channels: int, number of output channels corresponding
                           to the maximum number of boxes
        size:         int, size of input feature map
        """

        super(SmallNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d())

        self.fc = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(128*7*7, 60),
            nn.Linear(60, 1))
        self.out_act = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 128*7*7)
        out = self.fc(out)
        out = self.out_act(out)
        if self.training:
            return out
        else:
            return out.max(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
