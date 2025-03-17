import torch.nn as nn

class BaseCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_stride=1, pool_kernel=(1, 1), pool_stride=(1, 1)):
        super(BaseCNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=conv_stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class BaseCNNModel(nn.Module):
    def __init__(self):
        super(BaseCNNModel, self).__init__()

        self.block1 = BaseCNNBlock(in_channels=1, out_channels=32, pool_kernel=(3, 3), pool_stride=(1, 1))
        self.block2 = BaseCNNBlock(in_channels=32, out_channels=64, pool_kernel=(4, 4), pool_stride=(2, 2))
        self.block3 = BaseCNNBlock(in_channels=64, out_channels=128, pool_kernel=(4, 4), pool_stride=(2, 2))
        self.block4 = BaseCNNBlock(in_channels=128, out_channels=256, pool_kernel=(3, 3), pool_stride=(3, 3))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7680, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        return x
