
import numpy as np
import random
import shutil
from tqdm import tqdm
import os

test_path = os.path.join('/home/users/s19013225/workspace/data/test')
train_aug_path = os.path.join('/home/users/s19013225/workspace/data/train_augmented')

class_name_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

for class_name in class_name_list:
    test_class_path = os.path.join(test_path, class_name)
    test_class_img_list = os.listdir(test_class_path)
    
    train_class_path = os.path.join(train_aug_path, class_name)
    train_class_img_list = os.listdir(train_class_path)

    # check 용
    print("\n\n")
    print(">> check : {}".format(class_name))
    print("train_aug : ", len(train_class_img_list))
    print("test(not aug) : ", len(test_class_img_list))
    print("all : ", len(train_class_img_list) + len(test_class_img_list))
    
"""
>> check : akiec
train_aug :  6463
test(not aug) :  33
all :  6496

>> check : bcc
train_aug :  6464
test(not aug) :  52
all :  6516

>> check : bkl
train_aug :  6918
test(not aug) :  110
all :  7028

>> check : df
train_aug :  6061
test(not aug) :  12
all :  6073


train_aug :  7006
test(not aug) :  112
all :  7118

>> check : nv
train_aug :  6034
test(not aug) :  671
all :  6705

>> check : vasc
train_aug :  6085
test(not aug) :  15
all :  6100

"""