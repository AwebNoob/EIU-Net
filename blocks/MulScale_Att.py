import torch
import torch.nn as nn
import torch.nn.functional as F


def xy_avg_max(x_y: list):
    res_avg = []
    res_max = []
    for idx in x_y:
        avg_pool = F.adaptive_avg_pool2d(idx, 1)
        max_pool = F.adaptive_max_pool2d(idx, 1)
        res_avg.append(avg_pool)
        res_max.append(max_pool)
    res = res_avg + res_max
    return torch.cat(res, dim=1)


class ChannelAtt(nn.Module):
    def __init__(self, channels):
        super(ChannelAtt, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, momentum=0.9, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, x_ch, y_ch, out_ch, resize_mode='bilinear'):
        super(MultiScaleAttention, self).__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(x_ch, y_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(y_ch, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(y_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.resize_mode = resize_mode

        self.conv_xy = nn.Sequential(
            nn.Conv2d(4 * y_ch, y_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch // 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(y_ch // 2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch, momentum=0.9),
        )
        self.channel_att = ChannelAtt(channels=y_ch)

    def prepare(self, x, y):
        x = self.prepare_x(x)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_expand = F.interpolate(y, x.shape[2:], mode=self.resize_mode, align_corners=True)
        return y_expand

    def fuse(self, x, y):
        attention = xy_avg_max([x, y])
        attention = self.channel_att(self.conv_xy(attention))

        out = x * attention + y * (1 - attention)
        out = self.conv_out(out)
        return out

    def forward(self, x, y):

        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

