import torch
import model
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
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = model.device

if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 128
    train_dataset = datasets.CIFAR10(root=r'/home/zhaoll/zhao/cifar/data', train=True, transform=transform_train,
                                     download=True)
    test_dataset = datasets.CIFAR10(root=r'/home/zhaoll/zhao/cifar/data', train=False, transform=transform_test,
                                    download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    transform = transforms.Compose([transforms.ToTensor()])
    trainLoss = 0.0
    start_epoch = 1
    test_loss_list = []
    train_loss_list = []
    acc_list = []
    epoches = 30

    model = model.CNN()
    model = model.to(device)

    # model.load_state_dict(torch.load('cifar_model_e_25.pth'))

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数 （内置了softmax）
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=200)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + epoches):
        train_N = 0.
        train_n = 0.
        trainLoss = 0.
        model.train()
        for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
            train_n = len(trainX)
            train_N += train_n
            # 分类网络
            #trainX = trainX.to(device)
            noise_image = trainX + torch.randn(trainX.shape) * 100 / 255
            noise_image = noise_image.to(device)
            trainY = trainY.to(device).long()
            optimizer.zero_grad()
            predY = model(noise_image)
            loss = criterion(predY, trainY)
            loss.backward()
            optimizer.step()
            trainLoss += loss.detach().cpu().numpy()
        trainLoss /= train_N
        train_loss_list.append(trainLoss)
        print('epoch:{}  trainLoss:{} '.format(epoch, trainLoss))
        # if(epoch % 5 ==0):
        test_N = 0.
        testLoss = 0.  # 网络测试损失
        correct = 0.
        model.eval()
        for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
            test_n = len(testX)
            test_N += test_n
            #testX = testX.to(device)
            test_noise_image = testX + torch.randn(testX.shape) * 100 / 255
            test_noise_image = test_noise_image.to(device)
            testY = testY.to(device).long()
            with torch.no_grad():
                # 真实分类网络测试损失
                test_predY = model(test_noise_image)
                loss = criterion(test_predY, testY)
                testLoss += loss.detach().cpu().numpy()
                _, predicted_x = torch.max(test_predY.data, 1)
                correct += (predicted_x == testY).sum()
        testLoss /= test_N
        test_loss_list.append(testLoss)
        acc = correct / test_N
        acc_list.append(acc)
        print(
            'epoch:{} testLoss:{} acc:{}'.format(epoch, testLoss,  acc))
        # scheduler_2.step()
        # scheduler.step()

    torch.save(model.state_dict(), 'cifar75.pth')

    # 数据保存
    file = open('data.txt', 'w')
    file.write('acc_list:' + str(acc_list) + '\n')
    file.close()






