
import numpy as np
import random
import shutil
from tqdm import tqdm
import os

# ransom.sample
# list(set(lst1).difference(lst2))
# shutil.move(path, newPath)

old_path = os.path.join('/home/users/s19013225/workspace/data/reorganized')
new_train_path = os.path.join('/home/users/s19013225/workspace/data/train')
new_test_path = os.path.join('/home/users/s19013225/workspace/data/test')

# 시도용
# akiec class
# akiec_path = '/home/users/s19013225/workspace/data/reorganized/akiec'
# akiec = os.listdir(akiec_path)
# akiec_train = random.sample(akiec, int(len(akiec)*0.9))
# akiec_test = list(set(akiec).difference(akiec_train))

# for img in tqdm(akiec_train):
#     shutil.copy(os.path.join(old_path, 'akiec', img), os.path.join(new_train_path, 'akiec', img))

# for img in tqdm(akiec_test):
#     shutil.copy(os.path.join(old_path, 'akiec', img), os.path.join(new_test_path, 'akiec', img))

# 자동화 버전
class_name_list = ['bcc', 'bkl', 'df', 'mel', 'vasc', 'nv'] # akiec는 이미 했기 때문에 제외함!

for class_name in class_name_list:
    class_path = os.path.join(old_path, class_name)
    class_img_list = os.listdir(class_path)
    class_train_list = random.sample(class_img_list, int(len(class_img_list)*0.9))
    class_test_list = list(set(class_img_list).difference(class_train_list))

    for img in tqdm(class_train_list):
        shutil.copy(os.path.join(old_path, class_name, img), os.path.join(new_train_path, class_name, img))

    for img in tqdm(class_test_list):
        shutil.copy(os.path.join(old_path, class_name, img), os.path.join(new_test_path, class_name, img))

    # check 용
    print(">> check : {}".format(class_name))
    print("all : ", len(class_img_list))
    print("train : ", len(class_train_list))
    print("test : ", len(class_test_list))
    print(len(class_img_list), " : ",len(class_train_list) + len(class_test_list))

    """
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 462/462 [00:00<00:00, 2989.42it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 3101.41it/s]
>> check : bcc
all :  514
train :  462
test :  52
514  :  514
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 989/989 [00:00<00:00, 2775.15it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:00<00:00, 2801.21it/s]
>> check : bkl
all :  1099
train :  989
test :  110
1099  :  1099
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2763.10it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 3097.52it/s]
>> check : df
all :  115
train :  103
test :  12
115  :  115
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1001/1001 [00:00<00:00, 2722.72it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 2710.72it/s]
>> check : mel
all :  1113
train :  1001
test :  112
1113  :  1113
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 127/127 [00:00<00:00, 2867.23it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 2773.89it/s]
>> check : vasc
all :  142
train :  127
test :  15
142  :  142
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6034/6034 [00:02<00:00, 2542.90it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 671/671 [00:00<00:00, 2820.41it/s]
>> check : nv
all :  6705
train :  6034
test :  671
6705  :  6705
    """