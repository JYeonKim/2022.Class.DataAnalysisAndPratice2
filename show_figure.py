import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import numpy as np

origin_data_path = os.path.join('/home/users/s19013225/workspace/data/reorganized')
class_name_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_img_len = []

for class_name in class_name_list:
    class_path = os.path.join(origin_data_path, class_name)
    class_img_list = os.listdir(class_path)
    class_img_len.append(len(class_img_list))

# check 용
print("all(원본) : ", class_img_len)
print("all(sum)", sum(class_img_len))

df = pd.DataFrame({'class': class_name_list, 'count':class_img_len})
df['percent'] = round(df['count'] / sum(class_img_len) * 100, 2)

# check 용
print(df)

fig = plt.figure(figsize=(10,10), dpi=200)
ax1 = fig.add_subplot(1, 1, 1)
ax1.bar(df['class'], df['count'])
for i in range(len(df['class'])):
    ax1.text(df['class'][i], df['count'][i]+20, str(df['percent'][i])+'%', horizontalalignment='center', fontsize=15)
    # ax1.text(df['class'][i], df['count'][i]/2, str(df['count'][i]), horizontalalignment='center', fontsize=15)
ax1.set_xlabel('Class', fontsize = 18)
ax1.set_ylabel('Number of image', fontsize = 18)
ax1.set_title("HAM10000 Dataset", fontsize = 20)
ax1.tick_params(axis = 'x', labelsize = 15)
fig.savefig('./figure/HAM1000 Dataset summary.png')
