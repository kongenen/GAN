import PIL.Image as Image
import os
from torchvision import transforms as transforms
import csv
fh = open(r'D:\代码\论文\me\data1.csv', 'r')
reader = csv.reader(fh)
#print(len(reader))
outfile = r'D:\代码\论文\test'
i = 401

for row in reader:
    im = Image.open(row[0])
    new_im = transforms.RandomHorizontalFlip(p=1)(im) #shuiping
    path = 'test_'+ '000' + str(i) + '.png'
    new_im.save(os.path.join(outfile, path))

    new_im = transforms.RandomVerticalFlip(p=1)(new_im)  # shuiping
    path = 'test_' + '0000' + str(i) + '.png'
    new_im.save(os.path.join(outfile, path))

    new_im = transforms.RandomVerticalFlip(p=1)(im) #chuizhi
    path = 'test_' + '00000' + str(i) + '.png'
    new_im.save(os.path.join(outfile, path))
    i +=1

