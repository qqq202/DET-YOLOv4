# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:48:48 2021

@author: YaoYee
"""

import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# path = "VOCdevkit/VOC2007/Annotations"
# path ='F:\yolov4-pytorch-master-UCAS-AOD\VOCdevkit\VOC2007\Annotations'
path = 'F:\yolov4-pytorch-master-DOTA-new\VOCdevkit\VOC2007\Annotations'
files = os.listdir(path)

area_list = []
ratio_list = []


def file_extension(path):
    return os.path.splitext(path)[1]


for xmlFile in tqdm(files, desc='Processing'):
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            tree = et.parse(os.path.join(path, xmlFile))
            root = tree.getroot()
            filename = root.find('filename').text
            # print("--Filename is", xmlFile)

            for Object in root.findall('object'):
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
                area_list.append(area)
                # print("Area is", area)

                ratio = (int(ymax) - int(ymin)) / (int(xmax) - int(xmin))
                ratio_list.append(ratio)
                # print("Ratio is", round(ratio,2))

square_array = np.array(area_list)  # 面积数组
square_max = np.max(square_array)
square_min = np.min(square_array)
square_mean = np.mean(square_array)
square_var = np.var(square_array)
# 计算取值范围内数量 # COCO中 small《32*32《medium《96*96《large
print('总数：',len(square_array))
mask = (square_array <= 10*10)
print('超小目标：', np.count_nonzero(square_array * mask))
mask = (square_array >= 10*10) & (square_array <= 50*50)
# mask = (square_array <= 2500)
# square_array * mask
print('小目标：', np.count_nonzero(square_array * mask))
mask = (square_array > 2500) & (square_array <= 90000)
print('中目标：', np.count_nonzero(square_array * mask))
mask = (square_array > 300*300)
print('大目标：', np.count_nonzero(square_array * mask))
# 或
# np.extract(mask, square_array)
# len(np.extract(mask, square_array))

plt.figure(1)
plt.hist(square_array, 20)
plt.xlabel('Area in pixel')
plt.ylabel('Frequency of area')
plt.title('Area\n' \
          + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
          + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
          )
plt.show()

ratio_array = np.array(ratio_list)
ratio_max = np.max(ratio_array)
ratio_min = np.min(ratio_array)
ratio_mean = np.mean(ratio_array)
ratio_var = np.var(ratio_array)
plt.figure(2)
plt.hist(ratio_array, 20)
plt.xlabel('Ratio of length / width')
plt.ylabel('Frequency of ratio')
plt.title('Ratio\n' \
          + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
          + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
          )
