from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import build_backbone
from mmcls.models.backbones import vision_transformer
from torchsummary import summary
from torchinfo import summary

def vit_b_16(pretrained):
    cfg = dict(type='VisionTransformer', arch='base')
    model = build_backbone(cfg)
    resume = '../model_data/mae_pretrain_vit_base.pth'

    if pretrained:
        model_dict = model.state_dict()
        # print(model_dict.keys())
        checkpoint = torch.load(resume)
        # for k, v in checkpoint['model'].items():
        #     print("keys:",k)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k[:6] == 'blocks':
                name = 'layers' + k[6:]
                if k[9:14]== 'norm1':
                    name = name[:9]+ 'ln1' + name[14:]
                elif k[9:14]== 'norm2':
                    name = name[:9]+ 'ln2' + name[14:]
                elif k[9:16]=='mlp.fc1':
                    name = name[:9] + 'ffn.layers.0.0' + name[16:]
                elif k[9:16] == 'mlp.fc2':
                    name = name[:9] + 'ffn.layers.1' + name[16:]
                elif k[10:15]== 'norm1':
                    name = name[:10]+ 'ln1' + name[15:]
                elif k[10:15]== 'norm2':
                    name = name[:10]+ 'ln2' + name[15:]
                elif k[10:17]=='mlp.fc1':
                    name = name[:10] + 'ffn.layers.0.0' + name[17:]
                elif k[10:17] == 'mlp.fc2':
                    name = name[:10] + 'ffn.layers.1' + name[17:]
            elif k[:4] == 'norm':
                name = 'ln1' + k[4:]
            elif k[:16] == 'patch_embed.proj':
                name = 'patch_embed.projection' + k[16:]
            else:
                name = k  # backbone字段在最前面，从第8个字符开始就可以去掉backbone
            new_state_dict[name] = v  # 新字典的key值对应的value一一对应
        model_dict.update(new_state_dict) # 更新现有的model_dict
        model.load_state_dict(model_dict) # 加载我们真正需要的state_dict
    return model


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
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,  padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

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

class ViTC(nn.Module):
    def __init__(self):
        super(ViTC, self).__init__()
        self.backbone = vit_b_16(pretrained=True)
        self.Cov1 = BasicConv(768,512)
        self.Up1 = Upsample(512,256)
        self.down_sample1 = conv2d(512, 1024, 3, stride=2)

    def forward(self, x):
        out, cls_token = self.backbone(x)[-1]
        out4 = self.Cov1(out)
        out3 = self.Up1(out4)
        out5 = self.down_sample1(out4)

        return out3, out4, out5

def ViTC_b_16(pretrained):
    model = ViTC()
    if pretrained:
        pass
    return model