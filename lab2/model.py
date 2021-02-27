import torch
import torch.nn as nn

from dataset import HEIGHT, WIDTH


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         '''
#         Checkpoint 2:
#             Use Sequential to rewrite Basic Block
#         '''

#     def forward(self, x):
#         '''
#         Checkpoint 2:
#             Use Sequential to rewrite Basic Block
#         '''
#         return x


class ConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            BasicBlock(in_channels, out_channels),
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x):
        return self.main(x)


# class ConvBlocks(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block1 = BasicBlock(in_channels, out_channels)
#         self.block2 = BasicBlock(out_channels, out_channels)
#         self.block3 = BasicBlock(out_channels, out_channels)
#         self.pooling = nn.MaxPool2d((2, 2))

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.pooling(x)
#         return x


class LPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlocks(3, 32),        # [batch, HEIGHT / 2, WIDTH / 2, 32]
            ConvBlocks(32, 64),       # [batch, HEIGHT / 4, WIDTH / 4, 64]
            ConvBlocks(64, 128),      # [batch, HEIGHT / 8, WIDTH / 8, 128]
            nn.Flatten(start_dim=1),  # [batch, HEIGHT / 8 x WIDTH / 8 x 128]
        )
        self.classifiers = nn.ModuleList()
        for _ in range(7):
            classifier = nn.Sequential(
                nn.Linear(HEIGHT // 8 * WIDTH // 8 * 128, 128),  # [batch, 128]
                nn.ReLU(),
                nn.Linear(128, 36),                              # [batch, 36]
            )
            self.classifiers.append(classifier)

    def forward(self, x):
        x = self.encoder(x)
        outputs = []
        for classifier in self.classifiers:
            output = classifier(x)          # [batch, 36]
            output = output.unsqueeze(1)    # [batch, 1, 36]
            outputs.append(output)
        '''
        Checkpoint 3:
            Concatenate a list of tensors `outputs` along the axis 1.
            The concatenated output should be in shape [batch, 7, 36].
        output = torch.cat(????)
        '''
        output = torch.cat(outputs, dim=1)
        return output


if __name__ == '__main__':
    model = ConvBlocks(3, 16)
    x = model(torch.randn((8, 3, 48, 48)))
    print(x.shape)
