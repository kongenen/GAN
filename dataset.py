#coding=utf-8
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import random
import csv
from PIL import Image



def createImgIndex(dataPath, ratio):
    # '''
    # 读取目录下面的图片制作包含图片信息、图片label的train.txt和val.txt
    # dataPath: 图片目录路径
    # ratio: val占比
    # return：label列表
    # '''
    fileList = os.listdir(dataPath)
    print(fileList)

    #print(fileList[0])   #test_001.png
    #random.shuffle(fileList)
    classList = []  # label列表
    # val 数据集制作
    with open('test1.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for i in range(int(len(fileList) * ratio)):
            row = []
            if '.png' in fileList[i]:
                fileInfo = fileList[i].split('_')   #将名称按 _ 分割    #['test', '206.png']
                #print(fileInfo)
                sectionName = fileInfo[0] + '_' + fileInfo[1]  # 切面名+标准与否
                row.append(os.path.join(dataPath, fileList[i]))  # 图片路径
                if sectionName not in classList:
                    classList.append(sectionName)
                row.append(classList.index(sectionName))
                writer.writerow(row)
        f.close()
    #train 数据集制作
    # with open('train_section1015.csv', 'w',newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(int(len(fileList) * ratio) + 1, len(fileList)):
    #         row = []
    #         if '.png' in fileList[i]:
    #             fileInfo = fileList[i].split('_')
    #             sectionName = fileInfo[0] + '_' + fileInfo[1]  # 切面名+标准与否
    #             row.append(os.path.join(dataPath, fileList[i]))  # 图片路径
    #             if sectionName not in classList:
    #                 classList.append(sectionName)
    #             row.append(classList.index(sectionName))
    #             writer.writerow(row)
    #     f.close()
    # print(classList, len(classList))
    # return classList


def default_loader(path):
    '''定义读取文件的格式'''
    return Image.open(path).resize((128, 128), Image.ANTIALIAS).convert('L')


class MyDataset(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r',encoding='utf-8')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        reader = csv.reader(fh)
        imgs = []
        for row in reader:
            imgs.append((row[0], int(row[1])))  # (图片信息，lable)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中row[0]和row[1]的信息
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.imgs)


if __name__ == '__main__':
    dirPath = r'/home/zhaoll/project/Gan/Demoise/test'   # 图片文件目录
    createImgIndex(dirPath, 1)                # 创建train.txt, val.txt

