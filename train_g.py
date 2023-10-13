#coding=utf-8
import torch
import model
import dataset
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import skimage
import noise
import copy
import attack



# plt.switch_backend('agg')

device = model.device

if __name__ == '__main__':

    def fgsm_attack(image,epsilon,data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image,0,1)
        return perturbed_image


    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    text_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((227, 227)),
        transforms.ToTensor(),
    ])

    train_data = dataset.MyDataset(txt='/home/zhaoll/project/Gan/Demoise/train.csv', transform=transforms.ToTensor())
    val_data = dataset.MyDataset(txt='/home/zhaoll/project/Gan/Demoise/test.csv', transform=transforms.ToTensor())
    test_data = dataset.MyDataset(txt='/home/zhaoll/project/Gan/Demoise/test.csv', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_data,batch_size=20,shuffle=False,num_workers=4)  #[1, 128, 128]



    start_epoch = 1
    test_loss_list = []
    train_loss_list = []
    train_d_loss_list = []
    epoches =20
    model_e = model.UNetModel()
    model_e = model_e.to(device)
    model_D = model.discriminator()
    model_D = model_D.to(device)
    model_e.load_state_dict(torch.load('model_oa_r_p5.pth'))
    model_D.load_state_dict(torch.load('dis_oa_r_p5.pth'))

    criterrion = nn.MSELoss()  # 是单目标平方差函数
    criterion_D = nn.BCELoss()  # 是单目标二分类交叉熵函数
    optimizer = optim.Adam(model_e.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model_e.parameters(), lr=1e-3,momentum=0.9)
    optimizer_D = optim.Adam(model_D.parameters(), lr=1e-5)


    #noise
    noisee = noise.noise1()
    noisee = noisee[1:129,1:129]
    noisee = torch.Tensor(noisee)
    noisee = torch.unsqueeze(noisee,0) #1,128,128


    #guass noise
    # noisee = torch.randn([128,128]) * 25 /255
    # noisee = torch.Tensor(noisee)
    # noisee = torch.unsqueeze(noisee,0) #1,128,128


    # 1,128,128
    for epoch in range(start_epoch, start_epoch + epoches):
        train_N = 0.
        train_n = 0.
        trainLoss = 0.
        trainloss_d = 0.
        model_e.train()
        model_D.train()
        for batch, [trainX, trainY] in enumerate(tqdm(train_loader, ncols=10)):
            train_n = len(trainX)
            train_N += train_n
            num = trainX.size(0)
            img= copy.deepcopy(trainX)

            #noise
            for i in range(len(trainX)):
                trainX[i] += noisee
            noise_image = trainX
            noise_image = noise_image.to(device)

            real_img = Variable(img).to(device)
            real_label = Variable(torch.ones(num)).to(device)
            fake_label = Variable(torch.zeros(num)).to(device)
            #判别网络
            #计算真实图片的损失
            real_out = model_D(real_img)
            real_out = real_out.squeeze(-1)
            d_loss_real = criterion_D(real_out,real_label)
            real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

            #PGD
            pgd_attack = attack.LinfPGDAttack(model=model_e, device=device, feat=None)
            noise_image = attack.perturb_batch(noise_image, real_img, model_e, pgd_attack)

            #计算生成图片的损失
            fake_img = model_e(noise_image)
            fake_img = Variable(fake_img).to(device)
            fake_out = model_D(fake_img)
            fake_out = fake_out.squeeze(-1)
            d_loss_fake = criterion_D(fake_out,fake_label)
            fake_scores = fake_out
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            trainloss_d += d_loss.detach().cpu().numpy()

            # FGLM
            # noise_image.requires_grad = True
            # output = model_e(noise_image)
            # optimizer.zero_grad()
            # loss = criterrion(output,real_img)
            # loss.backward()
            # data_grad = noise_image.grad.data
            # perturbed_data = fgsm_attack(noise_image,0.05,data_grad)
            # output = model_e(perturbed_data)

            #生成网络
            output = model_e(noise_image)
            optimizer.zero_grad()  # 梯度初始化为零
            loss = criterrion(output, real_img)
            loss.backward()  # get gradients on params  反向传播，计算当前梯度
            optimizer.step()  # SGD update  根据梯度更新网络参数
            trainLoss += loss.detach().cpu().numpy()  # .data放在cpu上  把tensor转换成numpy的格式
        trainLoss /= train_N
        trainloss_d /= train_N
        train_loss_list.append(trainLoss)
        train_d_loss_list.append(trainloss_d)
        print('epoch:{} trainloss_d:{} trainLoss:{} '.format(epoch,trainloss_d,trainLoss))
        #if(epoch % 5 ==0):
        test_N = 0.
        testLoss = 0. #生成网络测试损失
        model_e.eval()  # model.eval() 负责改变batchnorm、dropout的工作方式，如在eval()模式下，dropout是不工作的
        model_D.eval()
        for batch, [testX, testY] in enumerate(tqdm(test_loader, ncols=10)):
            test_n = len(testX)
            test_N += test_n
            test_img = copy.deepcopy(testX)
            test_img= test_img.to(device)

            #noise
            for i in range(len(testX)):
                testX[i] += noisee
            test_noise_image = testX
            test_noise_image = test_noise_image.to(device)

            # pgd
            pgd_attack = attack.LinfPGDAttack(model=model_e, device=device, feat=None)
            test_noise_image = attack.perturb_batch(test_noise_image, test_img, model_e, pgd_attack)

            #生成网络测试损失
            with torch.no_grad():
                output1 = model_e(test_noise_image)
            loss = criterrion(output1,test_img)
            testLoss += loss.detach().cpu().numpy()
        testLoss /= test_N
        test_loss_list.append(testLoss)
        print('epoch:{} testLoss:{}'.format(epoch,testLoss))


        #time
        #test_N = 0.
        # for batch, [testX, testY] in enumerate(tqdm(test_loader, ncols=10)):
        #
        #
        #     # noise
        #     for i in range(len(testX)):
        #         testX[i] += noisee * 3
        #     test_noise_image = testX
        #     test_noise_image = test_noise_image.to(device)
        #     #print(test_noise_image.shape)
        #
        #     # 潔~_彈~P缾Q纾\派K设U彍~_失
        #     with torch.no_grad():
        #         torch.cuda.synchronize()
        #         start = time.time()
        #         #print('start', start)
        #         output1 = model_e(test_noise_image)
        #         torch.cuda.synchronize()
        #         end = time.time()
        #         #print('end', end)
        #         print('time', (end - start))

    
    torch.save(model_e.state_dict(), 'model_oa_r_p5.pth')
    torch.save(model_D.state_dict(),'dis_oa_r_p5.pth')

    #model_e.load_state_dict(torch.load('model_oa.pth'))
    for batch, [testX, testY] in enumerate(tqdm(test_loader, ncols=10)):
        test = testX[0]

        test_noise_image = test + noisee

        #real image
        test0 = test.permute(1, 2, 0)
        plt.imshow(test0,cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()

        # noise image
        test1 = test_noise_image.permute(1, 2, 0)
        plt.imshow(test1,cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()

        test_noise_image = test_noise_image.to(device)
        test_noise_image = torch.unsqueeze(test_noise_image, 0)
        test_noise_image = test_noise_image.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            output1 = model_e(test_noise_image)

        test2 = output1[0]
        test2 = test2.permute(1, 2, 0)
        test2 = test2.detach().cpu().numpy()

        # denoise image
        plt.imshow(test2,cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()
        break

    

    
    def psnra(img,imclean,data_range):
        Img = img.detach().cpu().numpy()
        Iclean = imclean.detach().cpu().numpy()
        PSNR = 0
        Img = np.swapaxes(Img, 1, 3)
        Iclean = np.swapaxes(Iclean, 1, 3)
        #print(Img.shape)
        #print(Img.shape[0])
        for i in range(Img.shape[0]):
            PSNR += skimage.measure.compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        return (PSNR/Img.shape[0])
    
    def ssima(img,imclean):
        Img = img.detach().cpu().numpy()
        Iclean = imclean.detach().cpu().numpy()
        SSIM = 0 
        Img = np.swapaxes(Img, 1, 3)
        Iclean = np.swapaxes(Iclean, 1, 3)
        for i in range(Img.shape[0]):
            SSIM += skimage.measure.compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], multichannel=True)
        return (SSIM/Img.shape[0])

    psnr_test = 0
    ssim_test = 0
    test_N=0

    for batch, [testX, testY] in enumerate(tqdm(test_loader, ncols=10)):
        test_n = len(testX)
        test_N += test_n
        re_img= copy.deepcopy(testX)
        re_img = re_img.to(device)

        for i in range(len(testX)):
            testX[i] += noisee
        noise_image = testX
        noise_image = noise_image.to(device)

        # FGLM
        # noise_image.requires_grad = True
        # output = model_e(noise_image)
        # optimizer.zero_grad()
        # loss = criterrion(output,re_img)
        # loss.backward()
        # data_grad = noise_image.grad.data
        # perturbed_data = fgsm_attack(noise_image,0.05,data_grad)
        # with torch.no_grad():
        #    output = model_e(perturbed_data)

        # PGD
        pgd_attack = attack.LinfPGDAttack(model=model_e, device=device, feat=None)
        noise_image = attack.perturb_batch(noise_image, re_img, model_e, pgd_attack)
        with torch.no_grad():
            output = model_e(noise_image)


        #无攻击
        # with torch.no_grad():
        #     output = model_e(noise_image)

        psnr = psnra(output,re_img,1.)
        ssim = ssima(output,re_img)
        print('test_psnr',psnr)
        print('test_ssim',ssim)
        psnr_test += psnr
        ssim_test += ssim
    print('test_psnr_test',psnr_test)
    print('test_ssim_test',ssim_test)
    # a = test_N / test_n
    # psnr_test /= a
    # ssim_test /= a
    # print('test_psnr_test',psnr_test)
    # print('test_ssim_test',ssim_test)


    # psnr_test = 0
    # ssim_test = 0
    # test_N=0
    #
    # for batch, [valX, valY] in enumerate(tqdm(val_loader, ncols=4)):
    #     test_n = len(valX)
    #     test_N += test_n
    #     val_re_img= copy.deepcopy(valX)
    #     val_re_img = val_re_img.to(device)
    #
    #     # real noise
    #     for i in range(len(valX)):
    #         valX[i] += noise
    #     noise_image = valX
    #     noise_image = noise_image.to(device)
    #
    #     # guass noise
    #     # noise_image = valX + torch.randn(valX.shape) * 25 / 255
    #     # noise_image = noise_image.to(device)
    #
    #     with torch.no_grad():
    #         output = model_e(noise_image)
    #
    #     psnr = psnra(output,val_re_img,1.)
    #     ssim = ssima(output,val_re_img)
    #     print('val_psnr',psnr)
    #     print('val_ssim',ssim)
    #     psnr_test += psnr
    #     ssim_test += ssim
    # print('val_psnr_test',psnr_test)
    # print('val_ssim_test',ssim_test)
    # a = test_N / test_n
    # psnr_test /= a
    # ssim_test /= a
    # print('val_psnr_test',psnr_test)
    # print('val_ssim_test',ssim_test)






