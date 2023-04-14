import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.InvertedResidual import InvertedResidual
from blocks.ASPP import ASPP
from blocks.Soft_pooling import downsample_soft
from blocks.EPSA import EPSABlock
from blocks.MulScale_Att import MultiScaleAttention
from blocks.scale_attention import scale_atten_convblock_softpool
from blocks.reshape import reshaped

from utils.init import *

__all__ = ['EIU_Net']

import warnings

warnings.filterwarnings('ignore')


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EIU_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EIU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [32, 64, 128, 256, 512]

        self.enc_input = ResEncoder(self.n_channels, filters[0])
        self.encoder_1 = InvertedResidual(filters[0], filters[1])

        self.encoder_2 = InvertedResidual(filters[1], filters[2])

        self.encoder_3 = InvertedResidual(filters[2], filters[3])

        self.encoder_4 = EPSABlock(filters[3], 128)
        self.downsample = downsample_soft()
        self.aspp = ASPP(filters[4], [6, 12, 18])

        self.decoder_4 = UP(filters[4], filters[3])
        self.double_conv_4 = DoubleConv(filters[4], filters[3])
        self.decoder_3 = UP(filters[3], filters[2])
        self.double_conv_3 = DoubleConv(filters[3], filters[2])
        self.decoder_2 = UP(filters[2], filters[1])
        self.double_conv_2 = DoubleConv(filters[2], filters[1])
        self.decoder_1 = UP(filters[1], filters[0])
        self.double_conv_1 = DoubleConv(filters[1], filters[0])

        self.reshape_4 = reshaped(in_size=256, out_size=4, scale_factor=(224, 320))
        self.reshape_3 = reshaped(in_size=128, out_size=4, scale_factor=(224, 320))
        self.reshape_2 = reshaped(in_size=64, out_size=4, scale_factor=(224, 320))
        self.reshape_1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.scale_att = scale_atten_convblock_softpool(in_size=16, out_size=4)
        # self.scale_att = scale_attention(16, 4)

        self.final = OutConv(4, self.n_classes)

        self.mul_scale_att_1 = MultiScaleAttention(32, 64, 64)
        self.mul_scale_att_2 = MultiScaleAttention(64, 128, 128)
        self.mul_scale_att_3 = MultiScaleAttention(128, 256, 256)

        # self.out = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        # initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)  # 32 [-1, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [-1, 32, 112, 160]
        enc_1 = self.encoder_1(down1)  # 64 [-1, 64, 112, 160]
        mid_attention_1 = self.downsample(self.mul_scale_att_1(enc_input, enc_1))
        down2 = self.downsample(enc_1)  # [-1, 64, 56, 80]
        enc_2 = self.encoder_2(down2)  # 128  [-1, 128, 56, 80]
        mid_attention_2 = self.downsample(self.mul_scale_att_2(enc_1, enc_2))
        down3 = self.downsample(enc_2)  # [-1, 128, 28, 40]
        enc_3 = self.encoder_3(down3)  # 256  [-1, 256, 28, 40]
        mid_attention_3 = self.downsample(self.mul_scale_att_3(enc_2, enc_3))
        down4 = self.downsample(enc_3)  # [-1, 256, 28, 40]
        enc_4 = self.encoder_4(down4)  # 512 [-1, 512, 14, 20]
        enc_4 = self.aspp(enc_4)  # ASPP [-1, 512, 14, 20]
        up4 = self.decoder_4(enc_4)  # 256
        concat_4 = torch.cat((mid_attention_3, up4), dim=1)
        up4 = self.double_conv_4(concat_4)
        up3 = self.decoder_3(up4)  # 128
        concat_3 = torch.cat((mid_attention_2, up3), dim=1)
        up3 = self.double_conv_3(concat_3)
        up2 = self.decoder_2(up3)  # 64
        concat_2 = torch.cat((mid_attention_1, up2), dim=1)
        up2 = self.double_conv_2(concat_2)
        up1 = self.decoder_1(up2)  # 32
        concat_1 = torch.cat((enc_input, up1), dim=1)
        up1 = self.double_conv_1(concat_1)

        dsv4 = self.reshape_4(up4)  # [16, 4, 224, 320]
        dsv3 = self.reshape_3(up3)
        dsv2 = self.reshape_2(up2)
        dsv1 = self.reshape_1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 320]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]

        final = self.final(out)  # 2

        # out = self.out(final)

        return final



