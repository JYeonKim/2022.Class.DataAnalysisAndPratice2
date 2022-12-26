from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import random
import argparse
import torch.utils.data as data
import shutil
from tqdm import tqdm

sdir = '/home/users/s19013225/workspace/data/train'

aug_dir = os.path.join(sdir,'augmented')

if os.path.isdir(aug_dir): # see if aug_dir exists if so remove it to get a clean slate
    shutil.rmtree(aug_dir)
os.mkdir(aug_dir) # make a new empty aug_dir

filepaths=[]
labels=[]

# iterate through original_images and create a dataframe of the form filepaths, labels

original_images_dir=os.path.join(sdir)

for klass in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'vasc']:
    os.mkdir(os.path.join(aug_dir,klass)) # make the class subdirectories in the aug_dir
    classpath=os.path.join(original_images_dir, klass) # get the path to the classes (benign and maligant)
    flist=os.listdir(classpath)# for each class the the list of files in the class    
    for f in flist:        
        fpath=os.path.join(classpath, f) # get the path to the file
        filepaths.append(fpath)
        labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')

df=pd.concat([Fseries, Lseries], axis=1) # create the dataframe
gen=ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
groups=df.groupby('labels') # group by class
gen_image_num = {'akiec':20, 'bcc':12, 'bkl':5, 'df':57, 'mel':5, 'vasc':46}
for label in df['labels'].unique():  # for every class               
    group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
    # import pdb;pdb.set_trace()
    img_path = group.filepaths.to_list()
    sample_count=len(group)   # determine how many samples there are in this class  
    aug_img_count=0
    target_dir=os.path.join(aug_dir, label)

    for i, path in enumerate(tqdm(img_path)):
        # import pdb;pdb.set_trace()
        img = load_img(path)  # PIL 이미지
        x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
        x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

        # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
        # 지정된 `preview/` 폴더에 저장합니다.
        img_cnt = 0
        prefix = 'aug_'+str(i)+'_'
        # import pdb;pdb.set_trace()
        for batch in gen.flow(x, batch_size=1,save_to_dir=target_dir, save_prefix=prefix, save_format='jpg'):
            img_cnt += 1
            if img_cnt > gen_image_num[label]:
                break  # 이미지 20장을 생성하고 마칩니다


import os

path = '/home/users/s19013225/workspace/data/train/augmented'

# # 서브 디렉토리 목록 출력
# for root, subdirs, files in os.walk(path):
   
#     for d in subdirs:
#         fullpath = root + '/' + d
#         print(fullpath)

# print()

# 서브 디렉토리별 파일 개수 출력
for root, subdirs, files in os.walk(path):
    if len(files) > 0:
        print(root, len(files))