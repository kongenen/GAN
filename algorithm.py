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

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    return device

device = get_device()

class UnOptimizedNoiseLayer(nn.Module):  #加噪声函数
    def __init__(self):
        super(UnOptimizedNoiseLayer, self).__init__()

    def forward(self, input):

        return input + torch.randn(input.shape, device=device) #返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义

class ConvNoiseFunction(Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        # layer_input, layer_sigma, layer_u = args
        layer_input, layer_sigma, layer_u = args
        layer_noise = torch.randn(size=layer_input.shape, device=device)

        layer_sigma = torch.abs(layer_sigma)

        output = layer_input + layer_noise * layer_sigma + layer_u
        # output = layer_input + layer_noise * layer_sigma
        ctx.save_for_backward(layer_input, layer_sigma, layer_noise * layer_sigma, layer_u)
        # ctx.save_for_backward(layer_input, layer_sigma, layer_noise * layer_sigma)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        # print(grad_output.shape)
        layer_input, layer_sigma, layer_noise, layer_u = ctx.saved_tensors
        # layer_input, layer_sigma, layer_noise = ctx.saved_tensors
        grad_input = grad_sigma = grad_u = None
        # print(grad_output.shape)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_output * layer_noise / layer_sigma
        if ctx.needs_input_grad[2]:
            grad_u = grad_output
        return grad_input, grad_sigma, grad_u


def convnoisefunction(layer_input, layer_sigma, layer_u):
    return ConvNoiseFunction(layer_input, layer_sigma, layer_u)


class OptimizedNoiseLayer(nn.Module):
    def __init__(self, input_features):
        super(OptimizedNoiseLayer, self).__init__()
        self.sigma = torch.Tensor(*input_features)
        #print(self.sigma)
        self.u = torch.Tensor(*input_features)
        self.sigma.requires_grad = True
        self.u.requires_grad = True
        # print(self.sigma.shape)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.sigma)
        nn.init.xavier_normal_(self.u)
        self.sigma = self.sigma.to(device)
        self.sigma = nn.Parameter(self.sigma, requires_grad=True)
        self.u = self.u.to(device)
        self.u = nn.Parameter(self.u, requires_grad=True)
        # print(self.sigma.shape)

    def forward(self, input, pure=False):
        if self.training:
            # self.u.data = torch.clamp(self.u.data, -1, 1)
            return ConvNoiseFunction.apply(input, self.sigma, self.u)
            # return ConvNoiseFunction.apply(input, self.sigma)
        else:
            if pure:
                return input
            else:
                return ConvNoiseFunction.apply(input, self.sigma, self.u)
