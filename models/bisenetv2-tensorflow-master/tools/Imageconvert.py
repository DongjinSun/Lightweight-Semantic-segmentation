from PIL import Image
import os
import numpy as np
import tensorflow.compat.v1 as tf
import tqdm

o_raw_path = './data/icg_drone/dataset/original/' # 원본 이미지 경로 ./data/icg_drone/dataset/original
o_data_path = './data/icg_drone/dataset/resize_original/'  # 저장할 이미지 경로
l_raw_path = './data/icg_drone/dataset/label/' # 원본 이미지 경로 ./data/icg_drone/dataset/original
l_data_path = './data/icg_drone/dataset/resize_label/'  # 저장할 이미지 경로
# resize 시작 --------------------


# 저장할 경로 없으면 생성
if not os.path.exists(o_data_path):
    os.mkdir(o_data_path)
if not os.path.exists(l_data_path):
    os.mkdir(l_data_path)
#원본 이미지 경로의 모든 이미지 list 지정
data_list = os.listdir(o_raw_path)
re_size = (2000,3000)
# 모든 이미지 resize 후 저장하기
for name in tqdm.tqdm(data_list):
    if name.rfind("jpg") == -1:
        continue
  # 이미지 열기
    im = Image.open(o_raw_path + name)
    # 이미지 resize
    im = np.array(im)
    im = im.reshape(1,4000,6000,3)
    im2 = tf.image.resize_nearest_neighbor(images = im, size = re_size)
    im2 = np.array(im2)
    im2 = im2.reshape(re_size[0],re_size[1],3)
    im2 = Image.fromarray(im2)
    # 이미지 JPG로 저장
    im2.save(o_data_path + name)
print('end ::: original')

#원본 이미지 경로의 모든 이미지 list 지정
data_list = os.listdir(l_raw_path)
# 모든 이미지 resize 후 저장하기
for name in tqdm.tqdm(data_list):
    if name.rfind("png") == -1:
        continue
  # 이미지 열기
    im = Image.open(l_raw_path + name)
    # 이미지 resize
    im = np.array(im)
    im = im.reshape(1,4000,6000,1)
    im2 = tf.image.resize_nearest_neighbor(images = im, size = re_size)
    im2 = np.array(im2)
    im2 = im2.reshape(re_size[0],re_size[1])
    im2 = Image.fromarray(im2)
    im2.save(l_data_path + name)
print('end ::: label')