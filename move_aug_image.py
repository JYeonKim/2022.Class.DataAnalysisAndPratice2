
import numpy as np
import random
import shutil
from tqdm import tqdm
import os

old_path = os.path.join('/home/users/s19013225/workspace/data/train/augmented')
new_train_path = os.path.join('/home/users/s19013225/workspace/data/train_augmented')

class_name_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'vasc'] # data augmented한 6개 class의 img만 이동

for class_name in class_name_list:
    class_path = os.path.join(old_path, class_name)
    class_img_list = os.listdir(class_path)

    for img in tqdm(class_img_list):
        shutil.copy(os.path.join(old_path, class_name, img), os.path.join(new_train_path, class_name, img))

    all_class_img_list = os.listdir(os.path.join(new_train_path, class_name))

    # check 용
    print(">> check : {}".format(class_name))
    print("augmented : ", len(class_img_list))
    print("all(원본 + augmented) : ", len(all_class_img_list))

"""
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6169/6169 [00:00<00:00, 7534.13it/s]
>> check : akiec
augmented :  6169
all(원본 + augmented) :  6463
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6002/6002 [00:00<00:00, 9849.76it/s]
>> check : bcc
augmented :  6002
all(원본 + augmented) :  6464
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5929/5929 [00:00<00:00, 9680.00it/s]
>> check : bkl
augmented :  5929
all(원본 + augmented) :  6918
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5958/5958 [00:00<00:00, 9736.16it/s]
>> check : df
augmented :  5958
all(원본 + augmented) :  6061
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6005/6005 [00:00<00:00, 8703.20it/s]
>> check : mel
augmented :  6005
all(원본 + augmented) :  7006
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5958/5958 [00:00<00:00, 10548.54it/s]
>> check : vasc
augmented :  5958
all(원본 + augmented) :  6085
"""