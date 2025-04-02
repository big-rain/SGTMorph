#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/14 19:05
# @Author  : ShengPengpeng
# @File    : retrieve.py
# @Description :


import pandas as pd
import numpy as np
import collections
import cv2
import glob
import os
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mping



path = 'cls_n7_sf_query.csv'
save_path = '../data/neuron7/res_sf/'


if not os.path.exists(save_path):
    os.makedirs(save_path)
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

df = pd.read_csv(path)

for index, row in df.iterrows():
    top_similarity = {}
    enbedding = row['embedding']
    enbedding = eval(enbedding)
    enbedding = np.array(enbedding)
    for index_0, row_0 in df.iterrows():
        if index_0 == index:
            continue
        else:
            enbedding_0 = row_0['embedding']
            enbedding_0 = eval(enbedding_0)
            enbedding_0 = np.array(enbedding_0)
            similarity = cosine_similarity(enbedding, enbedding_0)
            top_similarity[row_0['name']] = similarity
    sorted_list = sorted(top_similarity.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_list)

    ordered_dict = collections.OrderedDict(sorted_list)
    # top_k = dict(list(ordered_dict.items())[-5:])
    top_k = dict(list(ordered_dict.items())[:5])
    # print(top_k)
    fig_list = []
    fig_list.append(row['path'])
    top_name = []
    top_name.append(row['name'])
    plt.figure(figsize=(15, 15), dpi=500)
    for key_i, value_i in top_k.items():
        fig_path = df[df['name'] == key_i]
        for index_x, row_x in fig_path.iterrows():
            path_correponce = row_x['path']
        fig_list.append(path_correponce)
        top_name.append(key_i)
        # img = cv2.imread(image_path)

    for index, image_path in enumerate(fig_list):
        name_compare = top_name[index]  ###name
        if index == 0:
            value_similarity = 1
        else:
            value_similarity = top_k[name_compare] ##x
        level = index + 1

        # achor_path = row['path']
        print(image_path)
        img = cv2.imread(os.path.join(image_path))

        title = str(name_compare) +'--' + str(value_similarity)
        # 行，列，索引
        plt.subplot(6, 1, index+1)
        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.xticks([])
        plt.yticks([])

    # plt.show()
    save_path_concat = os.path.join(save_path, '%s'%(row['label']),'%s.png' %(row['name']))
    plt.savefig(save_path_concat)

    plt.close()