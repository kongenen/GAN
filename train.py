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
    trainLoss = 0.
    testLoss = 0.
    learning_rate = 0.0001
    start_epoch = 1
    test_loss_list = []
    test_c_loss_list = []
    test_x_loss_list = []
    train_loss_list = []
    train_d_loss_list = []
    train_c_loss_list = []
    acc_list = []
    acc_noise_list = []
    epoches = 30

    model_e = model.UNetModel()
    model_e = model_e.to(device)
    model_D = model.discriminator()
    model_D = model_D.to(device)
    model_C = model.CNN()
    model_C = model_C.to(device)

    # model_e.load_state_dict(torch.load('cifar_model_e_25.pth'))
    # model_D.load_state_dict(torch.load('cifar_model_D_25.pth'))
    # model_C.load_state_dict(torch.load('cifar_model_C_25.pth'))

    criterrion = nn.MSELoss()  # 是单目标平方差函数
    criterion_D = nn.BCELoss()  # 是单目标二分类交叉熵函数
    criterion_C = nn.CrossEntropyLoss()  # 交叉熵损失函数 （内置了softmax）
    optimizer = optim.Adam(model_e.parameters(), lr=0.001)
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.001)
    optimizer_C = optim.SGD(model_C.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
    # scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=200)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + epoches):
        train_N = 0.
        train_n = 0.
        trainLoss = 0.
        trainloss_d = 0.
        trainloss_c = 0.
        model_e.train()
        model_D.train()
        model_C.train()
        for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
            train_n = len(trainX)
            train_N += train_n
            noise_image = trainX + torch.randn(trainX.shape) * 25 / 255
            noise_image = noise_image.to(device)
            num = trainX.size(0)
            img = trainX
            real_img = Variable(img).to(device)
            real_label = Variable(torch.ones(num)).to(device)
            fake_label = Variable(torch.zeros(num)).to(device)
            # 判别网络
            # 计算真实图片的损失
            real_out = model_D(real_img)
            real_out = real_out.squeeze(-1)
            d_loss_real = criterion_D(real_out, real_label)
            # 计算生成图片的损失
            fake_img = model_e(noise_image)
            fake_img = Variable(fake_img).to(device)
            fake_out = model_D(fake_img)
            fake_out = fake_out.squeeze(-1)
            d_loss_fake = criterion_D(fake_out, fake_label)
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            trainloss_d += d_loss.detach().cpu().numpy()
            # 生成网络
            output = model_e(noise_image)
            trainX = trainX.to(device)
            loss = criterrion(output, trainX)
            optimizer.zero_grad()  # 梯度初始化为零
            loss.backward()  # get gradients on params  反向传播，计算当前梯度
            optimizer.step()  # SGD update  根据梯度更新网络参数
            trainLoss += loss.detach().cpu().numpy()  # .data放在cpu上  把tensor转换成numpy的格式
            # 分类网络
            trainY = trainY.to(device).long()
            optimizer_C.zero_grad()
            predY = model_C(trainX)
            loss = criterion_C(predY, trainY)
            loss.backward()
            optimizer_C.step()
            trainloss_c += loss.detach().cpu().numpy()
        trainLoss /= train_N
        trainloss_d /= train_N
        trainloss_c /= train_N
        train_loss_list.append(trainLoss)
        train_d_loss_list.append(trainloss_d)
        train_c_loss_list.append((trainloss_c))
        print('epoch:{} trainloss_d:{} trainLoss:{} trainloss_c:{}'.format(epoch, trainloss_d, trainLoss, trainloss_c))
        # if(epoch % 5 ==0):
        test_N = 0.
        testLoss = 0.  # 生成网络测试损失
        testloss_c = 0.  # 生成分类网络测试损失
        testloss_x = 0.  # 真实分类网络测试损失
        correct = 0.
        correct_x = 0.
        model_e.eval()
        model_D.eval()
        model_C.eval()
        for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
            test_n = len(testX)
            test_N += test_n
            test_noise_image = testX + torch.randn(testX.shape) * 25 / 255
            test_noise_image = test_noise_image.to(device)
            testX = testX.to(device)
            testY = testY.to(device).long()
            with torch.no_grad():
                # 生成网络测试损失
                output1 = model_e(test_noise_image)
                loss = criterrion(output1, testX)
                testLoss += loss.detach().cpu().numpy()
                # 生成分类网络测试损失
                # print(output1.shape)
                test_noise_predY = model_C(output1)
                # print(test_noise_predY.shape)
                loss = criterion_C(test_noise_predY, testY)
                testloss_c += loss.detach().cpu().numpy()
                _, predicted = torch.max(test_noise_predY.data, 1)
                correct += (predicted == testY).sum()
                # 真实分类网络测试损失
                test_predY = model_C(testX)
                loss = criterion_C(test_predY, testY)
                testloss_x += loss.detach().cpu().numpy()
                _, predicted_x = torch.max(test_predY.data, 1)
                correct_x += (predicted_x == testY).sum()
        testLoss /= test_N
        test_loss_list.append(testLoss)
        testloss_c /= test_N
        test_c_loss_list.append(testloss_c)
        testloss_x /= test_N
        test_x_loss_list.append(testloss_x)
        acc_noisc = correct / test_N
        acc_noise_list.append(acc_noisc)
        acc = correct_x / test_N
        acc_list.append(acc)
        print(
            'epoch:{} testLoss:{} testloss_c:{} testloss_x:{}  acc_noisc:{} acc:{}'.format(epoch, testLoss, testloss_c,
                                                                                           testloss_x, acc_noisc, acc))
        # scheduler_2.step()
        # scheduler.step()

    torch.save(model_e.state_dict(), 'cifar_model_e_25.pth')
    torch.save(model_D.state_dict(), 'cifar_model_D_25.pth')
    torch.save(model_C.state_dict(), 'cifar_model_C_25.pth')
    # 数据保存
    file = open('data_noise25.txt', 'w')
    file.write('acc_noise_list:' + str(acc_noise_list) + '\n')
    file.write('acc_list:' + str(acc_list) + '\n')
    file.close()






