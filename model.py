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
# import sys
#
# print(sys.version)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        print('aaaaaaaaaaaaaaaaaaaa')
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
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))


class Mp(nn.Module):
    def __init__(self, k=2):
        super(Mp, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Ellan(nn.Module):
    def __init__(self,c1,c2,k=3):
        super(Ellan, self).__init__()
        self.conv1 = Conv(c1, int(c2/4), k=3)
        self.conv2 = Conv(c1, int(c2/4), k=3)
        self.conv3 = Conv(int(c2/4), int(c2/4), k)
        self.conv4 = Conv(int(c2/4), int(c2/4), k)
        self.conv5 = Conv(int(c2/4), int(c2/4), k)
        self.conv6 = Conv(int(c2/4), int(c2/4), k)
        self.conv7 = Conv(c2, c2, k=1)


    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = torch.cat([x1, x2, x4, x6], dim=1)

        return self.conv7(x7)

class MP(nn.Module):
    def __init__(self,c1,c2):
        super(MP, self).__init__()
        self.conv1 = Conv(c1, int(c2/2), k=3)
        self.conv2 = Conv(int(c2/2), int(c2/2), k=3, s=2)
        self.mp = Mp()
        self.conv3 = Conv(c1, int(c2/2), k=1)


    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x3 = self.mp(x)
        x4 = self.conv3(x3)

        return torch.cat([x2, x4], dim=1)


class Ellan2(nn.Module):
    def __init__(self, c1, c2, k=3):
        super(Ellan2, self).__init__()
        self.conv1 = Conv(c1, c2, k=3)
        self.conv2 = Conv(c1, c2, k=3)
        self.conv3 = Conv(c2, c2, k)
        self.conv4 = Conv(c2, c2, k)
        self.conv5 = Conv(c2, c2, k)
        self.conv6 = Conv(c2, c2 , k)
        self.conv7 = Conv(int(c2 * 4), c2, k=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = torch.cat([x1, x2, x4, x6], dim=1)

        return self.conv7(x7)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = Conv(1, 32)
        self.conv2 = Conv(32, 64, k=3, s=2)
        self.ellan = Ellan(64, 128)
        self.noise_layer1 = OptimizedNoiseLayer([128,64,64])

        self.mp1 = MP(128, 128)
        self.ellan1 = Ellan(128, 256)
        self.noise_layer2 = OptimizedNoiseLayer([256, 32, 32])
        self.mp2 = MP(256, 256)
        self.ellan2 = Ellan(256, 512)
        self.noise_layer3 = OptimizedNoiseLayer([512, 16, 16])
        self.mp3 = MP(512, 512)
        self.ellan3 = Ellan(512, 1024)
        self.noise_layer4 = OptimizedNoiseLayer([1024, 8, 8])
        self.mp4 = MP(1024, 1024)
        self.ellan4 = Ellan(1024, 1024)
        self.noise_layer5 = OptimizedNoiseLayer([1024, 4, 4])

        self.conv3 = Conv(1024, 512, k=1)

        self.up = nn.Upsample(None, 2, 'nearest')
        self.conv5 = Conv(1024, 512, k=1)
        self.ellan5 = Ellan2(1024, 256)
        self.noise_layer6 = OptimizedNoiseLayer([256, 8, 8])
        self.conv6 = Conv(512, 256, k=1)
        self.ellan6 = Ellan2(512, 128)
        self.noise_layer7 = OptimizedNoiseLayer([128, 16, 16])
        self.conv7 = Conv(256, 128, k=1)
        self.ellan7 = Ellan2(256, 64)
        self.noise_layer8 = OptimizedNoiseLayer([64, 32, 32])
        self.conv8 = Conv(128, 64, k=1)
        self.ellan8 = Ellan2(128, 32)
        self.noise_layer9 = OptimizedNoiseLayer([32, 64, 64])
        self.ellan9 = Ellan2(64, 32)
        self.noise_layer10 = OptimizedNoiseLayer([32, 128, 128])
        self.conv9 = Conv(32, 1, k=1)

        self.conv1024 = Conv(1024, 1024, k=1)
        self.coatten1024 = CoordAttention(1024, 1024)
        self.conv512 = Conv(512, 512, k=1)
        self.coatten512 = CoordAttention(512, 512)
        self.conv256 = Conv(256, 256, k=1)
        self.coatten256 = CoordAttention(256, 256)
        self.conv128 = Conv(128, 128, k=1)
        self.coatten128 = CoordAttention(128, 128)
        self.conv64 = Conv(64, 64, k=1)
        self.coatten64= CoordAttention(64, 64)

    def forward(self, x):
        x0 = self.conv1(x)
        x00 = self.conv2(x0)
        x1 = self.noise_layer1(self.ellan(x00))              #128  320 320
        x2 = self.noise_layer2(self.ellan1(self.mp1(x1)))  #256  160 160
        x3 = self.noise_layer3(self.ellan2(self.mp2(x2)))  #512  80 80
        x4 = self.noise_layer4(self.ellan3(self.mp3(x3)))  #1024 40 40
        x5 = self.noise_layer5(self.ellan4(self.mp4(x4)))  #1024 20 20
        x6 = self.conv3(x5)  # 512 20 20
        x7 = self.noise_layer6(self.ellan5(self.coatten1024(self.conv1024(torch.cat([self.up(x6), self.conv5(x4)], dim=1)))))  #256 40 40
        x8 = self.noise_layer7(self.ellan6(self.coatten512(self.conv512(torch.cat([self.up(x7), self.conv6(x3)], dim=1)))))    #128 80 80
        x9 = self.noise_layer8(self.ellan7(self.coatten256(self.conv256(torch.cat([self.up(x8), self.conv7(x2)], dim=1)))))    #64 160 160
        x10 = self.noise_layer9(self.ellan8(self.coatten128(self.conv128(torch.cat([self.up(x9), self.conv8(x1)], dim=1)))))   #32 320 320
        x11 = self.noise_layer10(self.ellan9(self.coatten64(self.conv64(torch.cat([self.up(x10), x0], dim=1)))))                #32 640 640
        x12 = self.conv9(x11)  #1 640 640

        return [x12, x7, x8, x9, x10]

class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.ellan = Ellan(256, 512)
        self.ellan64 = Ellan2(64,64)
        self.conv1 = Conv(1, 32, k=3, s=2)      #32 /2
        self.ellan1 = Ellan(32, 64)             #64
        self.conv2 = Conv(64, 128, k=3, s=2)    #128 /4
        self.ellan2 = Ellan(128, 256)           #256
        self.conv3 = Conv(256, 512, k=3, s=2)   #512 /8
        self.ellan3 = Ellan2(512, 512)          #512
        self.conv4 = Conv(512, 512, k=3, s=2)   #512 /16
        self.ellan4 = Ellan(512, 1024)          #1024
        self.conv5 = Conv(1024, 1024, k=3, s=2) #1024 /32
        self.conv6 = Conv(1024, 256, k=1)
        self.dis = nn.Sequential(
            nn.Linear(4096, 1024),  # 输入特征数为2048，输出为1024
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(1024, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # sigmoid可以班实数映射到【0,1】，作为概率值，  # 多分类用softmax函数
        )

    def forward(self, x):

        if len(x[0])== 1:
            x1 = self.ellan1(self.conv1(x))
            x2 = self.ellan2(self.conv2(x1))
            x3 = self.ellan3(self.conv3(x2))
            x4 = self.ellan4(self.conv4(x3))
            x5 = self.conv6(self.conv5(x4))
            #print(x5.shape)
            x5 = x5.view(x.size(0), -1)
            x6 = self.dis(x5)
            #print(x6.shape)

        if len(x[0]) == 256:
            x3 = self.ellan(x)  #512 8 8
            x4 = self.ellan4(self.conv4(x3))
            x5 = self.conv6(x4)
            #print(x5.shape)
            x5 = x5.view(x.size(0), -1)
            x6 = self.dis(x5)

        if len(x[0]) == 128:
            x2 = self.ellan2(x)
            x3 = self.ellan3(self.conv3(x2))
            x4 = self.ellan4(self.conv4(x3))
            x5 = self.conv6(x4)
            #print(x5.shape)
            x5 = x5.view(x.size(0), -1)
            x6 = self.dis(x5)
            #print(x6.shape)

        if len(x[0])== 64:
            x2 = self.ellan64(x)
            x3 = self.ellan2(self.conv2(x2))
            x4 = self.ellan4(self.conv3(x3))
            x5 = self.conv6(self.conv5(x4))
            #print(x5.shape)
            x5 = x5.view(x.size(0), -1)
            x6 = self.dis(x5)
            #print(x6.shape)

        if len(x[0])== 32:
            x1 = self.ellan1(x)
            x2 = self.ellan2(self.conv2(x1))
            x3 = self.ellan3(self.conv3(x2))
            x4 = self.ellan4(self.conv4(x3))
            x5 = self.conv6(self.conv5(x4))
            #print(x5.shape)
            x5 = x5.view(x.size(0), -1)
            x6 = self.dis(x5)
            #print(x6.shape)

        return x6



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
        self.conv1_1 = Conv(1, 32)  #32,640,640
        self.resnet1 = ResBlockNFC_atten(32,32,(32,128,128))
        #self.resnet1 = ResBlockNFC(32, 32, (32, 128, 128))

        self.conv1_2 = Downconv(32,32)    #320,320
        self.conv2_1 = Conv(32, 64)
        self.resnet2 = ResBlockNFC_atten(64, 64, (64, 64, 64))
        #self.resnet2 = ResBlockNFC(64, 64, (64, 64, 64))

        self.conv2_2 = Downconv(64,64)  #160,160
        self.conv3_1 = Conv(64, 128)
        self.resnet3 = ResBlockNFC_atten(128, 128, (128, 32, 32))
        #self.resnet3 = ResBlockNFC(128, 128, (128, 32, 32))

        self.conv3_2 = Downconv(128, 128) #80,00
        self.conv4_1 = Conv(128, 256)
        self.resnet4 = ResBlockNFC_atten(256, 256, (256, 16, 16))
        #self.resnet4 = ResBlockNFC(256, 256, (256, 16, 16))

        self.conv4_2 = Downconv(256, 256)  #40,40
        self.conv5_1 = Conv(256,  512)
        self.resnet5 = ResBlockNFC_atten(512, 512, (512, 8, 8))
        #self.resnet5 = ResBlockNFC(512, 512, (512, 8, 8))

        self.conv5_2 = Downconv(512, 512)  #20,20
        self.convd = Conv(512, 1024)

        self.up6 = Up(1024, 512)         #40,40
        self.resnet6 = ResBlockNFC_atten(512, 512, (512, 8, 8))
        #self.resnet6 = ResBlockNFC(512, 512, (512, 8, 8))

        self.up7 = Up(512, 256)         #80,80
        self.resnet7 = ResBlockNFC_atten(256, 256, (256, 16, 16))
        #self.resnet7 = ResBlockNFC(256, 256, (256, 16, 16))

        self.up8 = Up(256, 128)         #160,160
        self.resnet8 = ResBlockNFC_atten(128, 128, (128, 32, 32))
        #self.resnet8 = ResBlockNFC(128, 128, (128, 32, 32))

        self.up9 = Up(128, 64)          #320,320
        self.resnet9 = ResBlockNFC_atten(64, 64, (64, 64, 64))
        #self.resnet9 = ResBlockNFC(64, 64, (64, 64, 64))

        self.up10 = Up(64, 32)         # 640,640
        self.resnet10 = ResBlockNFC_atten(32, 32, (32, 128, 128))
        #self.resnet10 = ResBlockNFC(32, 32, (32, 128, 128))

        self.conv = Conv(32,1)

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
        #print(x5_1.shape)

        x5_2 = self.conv5_2(x5_1)
        x = self.convd(x5_2)

        x = self.up6(x, x5_1)
        x6 = self.resnet6(x)

        x7 = self.up7(x6, x4_1)
        x7 = self.resnet7(x7)

        x8 = self.up8(x7, x3_1)
        x8 = self.resnet8(x8)

        x9 = self.up9(x8, x2_1)
        x9 = self.resnet9(x9)

        x10 = self.up10(x9,x1_1)
        x10 = self.resnet10(x10)

        output = self.conv(x10)

        return [output, x6, x7, x8, x9]


class UNetModel_u(nn.Module):
    def __init__(self):
        super(UNetModel_u, self).__init__()
        self.conv1_1 = Conv(1, 32)  #32,640,640
        self.resnet1 = ResBlock(32,32)
        #self.resnet1 = ResBlock_atten(32, 32)

        self.conv1_2 = Downconv(32,32)    #320,320
        self.conv2_1 = Conv(32, 64)
        self.resnet2 = ResBlock(64, 64)
        #self.resnet2 = ResBlock_atten(64, 64)

        self.conv2_2 = Downconv(64,64)  #160,160
        self.conv3_1 = Conv(64, 128)
        self.resnet3 = ResBlock(128, 128)
        #self.resnet3 = ResBlock_atten(128, 128)

        self.conv3_2 = Downconv(128, 128) #80,00
        self.conv4_1 = Conv(128, 256)
        self.resnet4 = ResBlock(256, 256)
        #self.resnet4 = ResBlock_atten(256, 256)

        self.conv4_2 = Downconv(256, 256)  #40,40
        self.conv5_1 = Conv(256,  512)
        self.resnet5 = ResBlock(512, 512)
        #self.resnet5 = ResBlock_atten(512, 512)

        self.conv5_2 = Downconv(512, 512)  #20,20
        self.convd = Conv(512, 1024)

        self.up6 = Up(1024, 512)         #40,40
        self.resnet6 = ResBlock(512, 512)
        #self.resnet6 = ResBlock_atten(512, 512)

        self.up7 = Up(512, 256)         #80,80
        self.resnet7 = ResBlock(256, 256)
        #self.resnet7 = ResBlock_atten(256, 256)

        self.up8 = Up(256, 128)         #160,160
        self.resnet8 = ResBlock(128, 128)
        #self.resnet8 = ResBlock_atten(128, 128)

        self.up9 = Up(128, 64)          #320,320
        self.resnet9 = ResBlock(64, 64)
        #self.resnet9 = ResBlock_atten(64, 64)

        self.up10 = Up(64, 32)         # 640,640
        self.resnet10 = ResBlock(32, 32)
        #self.resnet10 = ResBlock_atten(32, 32)

        self.conv = Conv(32,1)

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
        print(x5_1.shape)

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


class discriminator1(nn.Module):
    def __init__(self):
        super(discriminator1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            torch.nn.Conv2d(1, 32, 3, 1, 1),  # [32，128,128]
            torch.nn.Conv2d(32, 32, 3, 2, 1), # [32，64,64]
            #OptimizedNoiseLayer((32, 64, 64)),
            torch.nn.MaxPool2d(2),  # [32，32,32]
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),  # [64，32,32]
            torch.nn.Conv2d(64, 64, 3, 2, 1),   # [64，16,16]
            #OptimizedNoiseLayer((64, 16, 16)),
            torch.nn.MaxPool2d(2),  # [64，8,8]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),  # [128，8,8]
            torch.nn.Conv2d(128, 256, 3, 2, 1),  # [256，4,4]
            #OptimizedNoiseLayer((256, 4, 4)),
            torch.nn.MaxPool2d(2),  # [256，2,2]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()) # [256，2,2]
        self.dis = nn.Sequential(
            nn.Linear(1024, 512),  # 输入特征数为2048，输出为1024
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


if __name__ == '__main__':
    x = UNetModel()
    #z = Dis()
    y = torch.rand(3,1,128,128)
    print(y.shape)
    a = x(y)
    print(a.shape)

