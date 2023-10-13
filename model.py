import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABCMeta, abstractmethod
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import time
import pickle as pkl
from torch import nn
import torch.optim as optim
import torch.nn.init
import math
from algorithm import OptimizedNoiseLayer
from cooatten import CoordAttention

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    return device

device = get_device()


class ResBlockNFC(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, stride=1):
        super(ResBlockNFC, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.noise_layer = OptimizedNoiseLayer(output_shape)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.noise_layer(out)
        out = self.relu(out)

        return out
#加注意力机制
class ResBlockNFC_atten(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, stride=1):
        super(ResBlockNFC_atten, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.noise_layer = OptimizedNoiseLayer(output_shape)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.coatten = CoordAttention(out_channels,out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.coatten(x)
        out = self.conv3(x)
        out = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.noise_layer(out)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#加注意力机制
class ResBlock_atten(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_atten, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.coatten = CoordAttention(out_channels, out_channels)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.coatten(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,stride = 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Downconv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Downconv,self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.conv1_1 = Conv(3, 32)  #32,640,640
        self.resnet1 = ResBlockNFC_atten(32,32,(32,32,32))

        self.conv1_2 = Downconv(32,32)    #320,320
        self.conv2_1 = Conv(32, 64)
        self.resnet2 = ResBlockNFC_atten(64, 64, (64, 16, 16))

        self.conv2_2 = Downconv(64,64)  #160,160
        self.conv3_1 = Conv(64, 128)
        self.resnet3 = ResBlockNFC_atten(128, 128, (128, 8, 8))

        self.conv3_2 = Downconv(128, 128) #80,00
        self.conv4_1 = Conv(128, 256)
        self.resnet4 = ResBlockNFC_atten(256, 256, (256, 4, 4))

        self.conv4_2 = Downconv(256, 256)  #40,40
        self.conv5_1 = Conv(256,  512)
        self.resnet5 = ResBlockNFC_atten(512, 512, (512, 2, 2))

        self.conv5_2 = Downconv(512, 512)  #20,20
        self.convd = Conv(512, 1024)

        self.up6 = Up(1024, 512)         #40,40
        self.resnet6 = ResBlockNFC_atten(512, 512, (512, 2, 2))

        self.up7 = Up(512, 256)         #80,80
        self.resnet7 = ResBlockNFC_atten(256, 256, (256, 4, 4))

        self.up8 = Up(256, 128)         #160,160
        self.resnet8 = ResBlockNFC_atten(128, 128, (128, 8, 8))

        self.up9 = Up(128, 64)          #320,320
        self.resnet9 = ResBlockNFC_atten(64, 64, (64, 16, 16))

        self.up10 = Up(64, 32)         # 640,640
        self.resnet10 = ResBlockNFC_atten(32, 32, (32, 32, 32))

        self.conv = Conv(32,3)

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x1_1 = self.resnet1(x1_1)

        x1_2 = self.conv1_2(x1_1)
        x2_1 = self.conv2_1(x1_2)
        x2_1 = self.resnet2(x2_1)

        x2_2 = self.conv2_2(x2_1)
        x3_1 = self.conv3_1(x2_2)
        x3_1 = self.resnet3(x3_1)

        x3_2 = self.conv3_2(x3_1)
        x4_1 = self.conv4_1(x3_2)
        x4_1 = self.resnet4(x4_1)

        x4_2 = self.conv4_2(x4_1)
        x5_1 = self.conv5_1(x4_2)
        x5_1 = self.resnet5(x5_1)

        x5_2 = self.conv5_2(x5_1)
        x = self.convd(x5_2)

        x = self.up6(x, x5_1)
        x = self.resnet6(x)

        x = self.up7(x, x4_1)
        x = self.resnet7(x)

        x = self.up8(x, x3_1)
        x = self.resnet8(x)

        x = self.up9(x, x2_1)
        x = self.resnet9(x)

        x = self.up10(x,x1_1)
        x = self.resnet10(x)

        output = self.conv(x)

        return output

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            torch.nn.Conv2d(3, 32, 3, 1, 1),  # [32，128,128]
            torch.nn.Conv2d(32, 32, 3, 2, 1), # [32，64,64]
            OptimizedNoiseLayer((32, 16, 16)),
            #torch.nn.MaxPool2d(2),  # [32，32,32]
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),  # [64，32,32]
            torch.nn.Conv2d(64, 64, 3, 2, 1),   # [64，16,16]
            OptimizedNoiseLayer((64, 8, 8)),
            #torch.nn.MaxPool2d(2),  # [64，8,8]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),  # [128，8,8]
            torch.nn.Conv2d(128, 256, 3, 2, 1),  # [256，4,4]
            OptimizedNoiseLayer((256, 4, 4)),
            #torch.nn.MaxPool2d(2),  # [256，2,2]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()) # [256，2,2]
        self.dis = nn.Sequential(
            nn.Linear(4096, 512),  # 输入特征数为2048，输出为1024
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(512, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()   # sigmoid可以班实数映射到【0,1】，作为概率值，  # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # view函数相当于numpy的reshape
        x = self.dis(x)
        return x

class CNn(nn.Module):   #CNN
    def __init__(self, block):
        super(CNn, self).__init__()
        self.conv1 = torch.nn.Sequential(
            #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            torch.nn.Conv2d(3, 16, 3, 1, 1), #[16，32,32]
            torch.nn.Conv2d(16, 32, 3, 1, 1),#[32，32,32]
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 128, 3, 1, 1), #[128，16,16]
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 128, 128, 3, 1)  # (b, 128, 8, 8)
        # 第2，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 128, 256, 4, 2)  # (b, 256, 4, 4)
        # 第2，通道数*2，图片尺寸/2
        self.layer3 = self.make_layer(block, 256, 512, 4, 2)  # (b, 512, 2, 2)

        # [512，2,2]
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) #view函数相当于numpy的reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def CNN():
    return CNn(ResBlock)

# if __name__ == '__main__':
#     x = UNetModel()
#     w = discriminator()
#     y = torch.rand(3,1,128,128)
#     #print(y.shape)
#     a = w(y)
#     print(a.shape)
