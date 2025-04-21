import torch.nn as nn
import torch
import torch.nn as nn
import torch

class BaseCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_stride=1, pool_kernel=(1, 1), pool_stride=(1, 1)):
        super(BaseCNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=conv_stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.residual(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = x + identity
        x = self.relu(x)
        x = self.pool(x)
        return x

class BaseCNNModel(nn.Module):
    def __init__(self, l1_factor=0.00001, l2_factor=0.0001):
        super(BaseCNNModel, self).__init__()
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

        self.block1 = BaseCNNBlock(in_channels=1, out_channels=16, pool_kernel=(3, 3), pool_stride=(1, 1))
        self.block2 = BaseCNNBlock(in_channels=16, out_channels=32, pool_kernel=(4, 4), pool_stride=(2, 2))
        self.block3 = BaseCNNBlock(in_channels=32, out_channels=64, pool_kernel=(4, 4), pool_stride=(2, 2))
        self.block4 = BaseCNNBlock(in_channels=64, out_channels=128, pool_kernel=(3, 3), pool_stride=(3, 3))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3840, out_features=512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=512, out_features=64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=1)
        )

    def get_regularization_loss(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += (param ** 2).sum()
        return self.l1_factor * l1_loss + self.l2_factor * l2_loss

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Calculate padding to maintain same size
        effective_kernel_size = (kernel_size - 1) * dilation
        same_padding = effective_kernel_size // 2
        
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, (1, kernel_size), 
                              stride=stride, padding=(0, same_padding), dilation=(1, dilation))
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                              stride=stride, padding=(0, same_padding), dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_channels, kernel_size=3, dropout=0.2, l1_factor=0.00001, l2_factor=0.0001):
        super(TCNModel, self).__init__()
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation,
                    padding=(kernel_size-1) * dilation,
                    dropout=dropout
                )
            )
        self.temporal_blocks = nn.Sequential(*layers)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1] * 50 * 128, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.temporal_blocks(x)
        x = self.head(x)
        return x

    def get_regularization_loss(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += (param ** 2).sum()
        return self.l1_factor * l1_loss + self.l2_factor * l2_loss


