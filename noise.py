import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def noise1():
    img1 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/1.png")
    # print('img1_shape',img1.shape)
    # print('img1_min',np.min(img1))
    # print('img1_max',np.max(img1))
    img2 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/1_1.png")
    # print('img1_shape',img2.shape)
    # print('img2_min',np.min(img2))
    # print('img2_max',np.max(img2))
    img3 = img1-img2
    img3 = np.array(img3/255, dtype=float)
    # print('img3_shape',img3.shape)
    #print(img3)

    img2_1 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/2.png")
    img2_2 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/2_2.png")
    img2_3 = img2_1 - img2_2
    img2_3 = np.array(img2_3/255, dtype=float)
    #print(img2_3)

    img3_1 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/3.png")
    img3_2 = cv2.imread("/home/zhaoll/project/Gan/Demoise/noise/3_3.png")
    img3_3 = img3_1 - img3_2
    img3_3 = np.array(img3_3/255, dtype=float)
    #print(img3_3)

    noisee = (img3+img2_3+img3_3) / 3
    #print(noisee.shape)


    no1 = noisee[:, :, 0]
    #print(no1.shape)
    no2 = noisee[:, :, 1]
    #print(no2)
    no3 = noisee[:, :, 2]
    #print(no3)

    noi = (no1 + no2  + no3 ) / 3
    #print(noi)
    #print(noi.shape)

    return noi


    # plt.imshow(no1, cmap=plt.get_cmap('gray'))
    # plt.axis('off')
    # plt.show()
    # plt.imshow(no2, cmap=plt.get_cmap('gray'))
    # plt.axis('off')
    # plt.show()
    # plt.imshow(no3, cmap=plt.get_cmap('gray'))
    # plt.axis('off')
    # plt.show()
    # plt.imshow(noi, cmap=plt.get_cmap('gray'))
    # plt.axis('off')
    # plt.show()
    # plt.imshow(lena,cmap=plt.get_cmap('gray')

# if __name__ == '__main__':
#     noise1()