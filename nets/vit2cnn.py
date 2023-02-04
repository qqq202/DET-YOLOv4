import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from einops import rearrange
from einops.layers.torch import Rearrange
from torchsummary import summary

from nets.models_vit import vit_base_patch16, vit_b_16


def testvit2cnn():
    x = torch.Tensor(np.random.random(size=(1, 197, 768)))
    print('0', x.shape)  # 0 torch.Size([1, 197, 768])
    # ViT进行linear embed时
    # flatten: [B, C, H, W] -> [B, C, HW]  [-1, 768, 14, 14] [-1, 196, 768]
    # transpose: [B, C, HW] -> [B, HW, C]
    H = W = 14
    x = x[:, 1:H * W + 1, :]  # 舍弃C维度的第一个（即类别token），因为x = torch.cat((cls_tokens, x), dim=1)
    print('1', x.shape)  # 1 torch.Size([1, 196, 768])
    x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
    print('2', x.shape)  # 2 torch.Size([1, 768, 14, 14])
    x = rearrange(x, 'b c h w -> b (h w) c')
    print('3', x.shape)  # 3 torch.Size([1, 196, 768])


# -------------------------------------------------#
#   MISH激活函数
# -------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
# ---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BasicConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(BasicConv2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------#
#   可分离卷积
#   Conv2d + Conv2d
# ---------------------------------------------------#
class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = Mish()

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        out = self.bn(out)
        out = self.activation(out)
        return out


# ---------------------------------------------------#
#   上采样
#   Conv2d + Conv2d
# ---------------------------------------------------#
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# testvit2cnn()
# class CViTC_Head(nn.Module):
#     def __init__(self):
#         super(CViTC_Head, self).__init__()
#         self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)  # torch.Size([1, 32, 208, 208])
#         self.pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
#         self.dp_conv = DEPTHWISECONV(32, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)  # torch.Size([1, 32, 208, 208])
#         x = self.pad(x)  # torch.Size([1, 32, 224, 224])
#         x = self.dp_conv(x)  # torch.Size([1, 3, 224, 224])
#
#         return x

class CViTC_Head(nn.Module):
    def __init__(self):
        super(CViTC_Head, self).__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)  # torch.Size([1, 32, 208, 208])
        self.pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
        self.dp_conv = DEPTHWISECONV(32, 3)

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([1, 32, 208, 208])
        x = self.pad(x)  # torch.Size([1, 32, 224, 224])
        x = self.dp_conv(x)  # torch.Size([1, 3, 224, 224])

        return x

# class CViTC_Tail(nn.Module):
#     def __init__(self):
#         super(CViTC_Tail, self).__init__()
#         self.pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
#         self.dp_conv = DEPTHWISECONV(32, 3)
#         self.conv2 = BasicConv2(768, 1024, kernel_size=2, stride=1, padding=0)
#         self.up1 = Upsample(1024, 512)
#         self.up2 = Upsample(512, 256)
#
#     def forward(self, x):
#         H = W = 14
#         out = x[:, 1:H * W + 1, :]  # 舍弃C维度的第一个（即类别token），因为x = torch.cat((cls_tokens, x), dim=1)
#         # 1 torch.Size([1, 196, 768])
#         out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
#         # # 2 torch.Size([1, 768, 14, 14])
#         out5 = self.conv2(out)  # torch.Size([1, 1024, 13, 13])
#         out4 = self.up1(out5)  # torch.Size([1, 512, 26, 26])
#         out3 = self.up2(out4)  # torch.Size([1, 256, 52, 52])
#
#         return out3, out4, out5

class CViTC_Tail(nn.Module):
    def __init__(self):
        super(CViTC_Tail, self).__init__()
        self.pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
        self.dp_conv = DEPTHWISECONV(32, 3)
        self.conv2 = BasicConv2(768, 1024, kernel_size=2, stride=1, padding=0)
        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(512, 256)
        self.up3 = Upsample(1024, 512)

    def forward(self, x):
        H = W = 14
        out = x[:, 1:H * W + 1, :]  # 舍弃C维度的第一个（即类别token），因为x = torch.cat((cls_tokens, x), dim=1)
        # 1 torch.Size([1, 196, 768])
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        # # 2 torch.Size([1, 768, 14, 14])
        out5 = self.conv2(out)  # torch.Size([1, 1024, 13, 13])
        out4 = self.up1(out5)  # torch.Size([1, 512, 26, 26])
        out3 = self.up3(out5)  # torch.Size([1, 512, 26, 26])
        out3 = self.up2(out3)  # torch.Size([1, 256, 52, 52])

        return out3, out4, out5

# test=CViTC_Tail()
# print(summary(test.to('cuda'), (197, 768)))

class CViTC(nn.Module):
    def __init__(self):
        super(CViTC, self).__init__()
        self.head = CViTC_Head()
        self.backbone = vit_b_16(pretrained=True) # pretrained=True
        self.tail = CViTC_Tail()

    def forward(self, x):
        x = self.head(x)
        out = self.backbone(x)  # torch.Size([1, 1000]) torch.Size([1, 197, 768])
        out3, out4, out5 = self.tail(out)

        return out3, out4, out5
        # torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 1024, 13, 13])

def CViTC_b_16(pretrained):
    model = CViTC()
    if pretrained:
        pass
    return model

'''
class CViTC(nn.Module):
    def __init__(self):
        super(CViTC, self).__init__()
        self.backbone = vit_b_16(pretrained=True)
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)  # torch.Size([1, 32, 208, 208])
        self.pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
        self.dp_conv = DEPTHWISECONV(32, 3)
        self.conv2 = BasicConv2(768, 1024, kernel_size=2, stride=1, padding=0)
        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(512, 256)

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([1, 32, 208, 208])
        x = self.pad(x)  # torch.Size([1, 32, 224, 224])
        x = self.dp_conv(x)  # torch.Size([1, 3, 224, 224])
        #  backbone
        x, out = self.backbone(x)  # torch.Size([1, 1000]) torch.Size([1, 197, 768])
        H = W = 14
        out = out[:, 1:H * W + 1, :]  # 舍弃C维度的第一个（即类别token），因为x = torch.cat((cls_tokens, x), dim=1)
        # 1 torch.Size([1, 196, 768])
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        # # 2 torch.Size([1, 768, 14, 14])
        out5 = self.conv2(out)  # torch.Size([1, 1024, 13, 13])
        out4 = self.up1(out5)  # torch.Size([1, 512, 26, 26])
        out3 = self.up2(out4)  # torch.Size([1, 256, 52, 52])

        return out3, out4, out5
        # torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 1024, 13, 13])
'''

# input = torch.tensor(np.random.random(size=(1, 3, 416, 416)), dtype=torch.float)
# # conv1 = BasicConv(3, 32, kernel_size=3, stride=2)  # torch.Size([1, 32, 208, 208])
# # pad = nn.ZeroPad2d(padding=(8, 8, 8, 8))
# # conv2 = DEPTHWISECONV(32, 3)
# # x1 = conv1(input)
# # print(x1.shape)
# # x2 = pad(x1)
# # print(x2.shape)
# # x3 = conv2(x2)
# # print(x3.shape)
# model = CViTC()
# x1, x2, x3 = model(input)
# print(x1.shape, x2.shape, x3.shape)
# # print("Model's state_dict:")
# # for param_tensor in model.state_dict():
# #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# print(summary(model.to('cuda'), (3, 416, 416)))
# # print(summary(model.to('cuda'), (197, 768)))
