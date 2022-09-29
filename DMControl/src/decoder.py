# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Decoder(nn.Module):
    def __init__(self,
                #  obs_shape,
                 feature_dim,
                #  num_layers=2,
                 num_filters=32,
                 min_spat_size=11,
                 ):
        super().__init__()

        self.num_filters = num_filters
        self.min_spat_size = min_spat_size
        self.fc_expand = nn.Linear(feature_dim, num_filters * min_spat_size * min_spat_size)


        self.deconv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_filters*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=9, kernel_size=7, padding=0)
        )

    def forward(self, x):
        x = self.fc_expand(x)
        x = x.view(x.size(0), self.num_filters, self.min_spat_size, self.min_spat_size)
        x = self.deconv(x)
        x = (torch.tanh(x) + 1) / 2
        return x


if __name__ == '__main__':
    decoder = Decoder(feature_dim=50, min_spat_size=12, num_filters=32)
    x = torch.randn(2, 50)
    y = decoder(x)
    print(y.size())
