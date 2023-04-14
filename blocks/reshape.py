import torch
import torch.nn as nn


class reshaped(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(reshaped, self).__init__()
        self.reshape = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.reshape(input)
