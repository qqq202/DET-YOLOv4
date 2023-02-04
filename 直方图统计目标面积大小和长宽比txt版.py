# -*- coding: utf-8 -*-
"""
数据集格式
'imagesource':imagesource
'gsd':gsd
x 1, y 1, x 2, y 2, x 3, y 3, x 4, y 4, category, difficult
x 1, y 1, x 2, y 2, x 3, y 3, x 4, y 4, category, difficul
"""
import codecs
import os
import sys
import xml.etree.cElementTree as et
import shapely.geometry as shgeo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import Counter

# path = "VOCdevkit/VOC2007/Annotations"
# path ='F:\DOTA_labels'
path = 'F:\DOTA_qie_v2_labels'
files = os.listdir(path)

area_list = []
ratio_list = []
name_list = []


def file_extension(path):
    return os.path.splitext(path)[1]

def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            # poly = list(map(lambda x:np.array(x), object_struct['poly']))
            # object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # if (object_struct['long-axis'] < 15):
            #     object_struct['difficult'] = '1'
            #     global small_count
            #     small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects


for txtFile in tqdm(files, desc='Processing'):
    if not os.path.isdir(txtFile):
        if file_extension(txtFile) == '.txt':
            objects =parse_dota_poly(path+'/'+txtFile)
            for i in range(len(objects)): # 不能直接in列表
                area = objects[i]['area']
                name = objects[i]['name']
                print(area)
                area_list.append(area)
                # print("Area is", area)
                name_list.append(name)



# for xmlFile in tqdm(files, desc='Processing'):
#     if not os.path.isdir(xmlFile):
#         if file_extension(xmlFile) == '.xml':
#             tree = et.parse(os.path.join(path, xmlFile))
#             root = tree.getroot()
#             filename = root.find('filename').text
#             # print("--Filename is", xmlFile)
#
#             for Object in root.findall('object'):
#                 bndbox = Object.find('bndbox')
#                 xmin = bndbox.find('xmin').text
#                 ymin = bndbox.find('ymin').text
#                 xmax = bndbox.find('xmax').text
#                 ymax = bndbox.find('ymax').text
#
#                 area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
#                 area_list.append(area)
#                 # print("Area is", area)
#
#                 ratio = (int(ymax) - int(ymin)) / (int(xmax) - int(xmin))
#                 ratio_list.append(ratio)
#                 # print("Ratio is", round(ratio,2))

square_array = np.array(area_list)  # 面积数组
square_max = np.max(square_array)
square_min = np.min(square_array)
square_mean = np.mean(square_array)
square_var = np.var(square_array)
# 计算取值范围内数量 # COCO中 small《32*32《medium《96*96《large
print('总数：',len(square_array))
mask = (square_array <= 10*10)
print('超小目标：', np.count_nonzero(square_array * mask))
print('超小：{:.2%}'.format(np.count_nonzero(square_array * mask)/len(square_array)))
mask = (square_array >= 10*10) & (square_array <= 50*50)
# mask = (square_array <= 2500)
# square_array * mask
print('小目标：', np.count_nonzero(square_array * mask))
print('小：{:.2%}'.format(np.count_nonzero(square_array * mask)/len(square_array)))
mask = (square_array > 2500) & (square_array <= 90000)
print('中目标：', np.count_nonzero(square_array * mask))
print('中：{:.2%}'.format(np.count_nonzero(square_array * mask)/len(square_array)))
mask = (square_array > 300*300)
print('大目标：', np.count_nonzero(square_array * mask))
print('大：{:.2%}'.format(np.count_nonzero(square_array * mask)/len(square_array)))
# 或
# np.extract(mask, square_array)
# len(np.extract(mask, square_array))

print(Counter(name_list))


plt.figure(1)
plt.hist(square_array, 20)
plt.xlabel('Area in pixel')
plt.ylabel('Frequency of area')
plt.title('Area\n' \
          + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
          + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
          )
plt.show()

# ratio_array = np.array(ratio_list)
# ratio_max = np.max(ratio_array)
# ratio_min = np.min(ratio_array)
# ratio_mean = np.mean(ratio_array)
# ratio_var = np.var(ratio_array)
# plt.figure(2)
# plt.hist(ratio_array, 20)
# plt.xlabel('Ratio of length / width')
# plt.ylabel('Frequency of ratio')
# plt.title('Ratio\n' \
#           + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
#           + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
#           )
