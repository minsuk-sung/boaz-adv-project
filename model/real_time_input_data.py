#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2018/6/9 15:34
     # @Author  : Awiny
     # @Site    :
     # @Project : C3D-tensorflow
     # @File    : real_time_input_data.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning



import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

# 이미지 크롭하는 함수
def clip_images_to_tensor(video_imgs, num_frames_per_clip=16, crop_size=112):
    data = []
    # !! 질문 !! crop_mean.npy 에는 어떤 데이터가 담겨있나요?
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    tmp_data = video_imgs # 16 프레임 이미지들
    img_datas = []
    if(len(tmp_data)!=0): # 받아온 이미지들 없으면 안 됨 !
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8)) # 이미지 uint8 type으로 변환
        if img.width > img.height: # 너비가 더 길 경우
          scale = float(crop_size)/float(img.height)
          # cv2.resize(img, dsize) -> dsize : Manual Size, (가로,세로) 형태의 tuple
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else: # 세로 길이가 더 길 경우
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        # 위의 조건문 : 짧은 쪽 길이를 112로 맞춰서 그 비율에 맞게 resize
        # 밑 : 그리고 나서 crop 해주는게 아닐까 싶은데, 잘 모르겠다.
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
     # data.append(img_datas)

    # pad (duplicate) data/label if less than batch_size
    data.append(img_datas)

    np_arr_data = np.array(data).astype(np.float32)

    return np_arr_data # 112x112로 크롭한 이미지들 배열로 다시 리턴 
