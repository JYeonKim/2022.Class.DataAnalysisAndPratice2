import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
import numpy as np

# ver10
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver10_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver10_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver10_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver10_SGD_1e-3_epoch_100/test_acc.npy')

fig = plt.figure(figsize=(15,10), dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.arange(len(train_loss)), train_loss, label='train')
ax1.plot(np.arange(len(test_loss)), test_loss, label='test')
ax1.legend()
ax1.set_title("Loss",fontsize=16)
ax1.set_xlabel('Epoch', fontsize=16)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(len(train_acc)), train_acc, label='train')
ax2.plot(np.arange(len(test_acc)), test_acc, label='test')
ax2.legend()
ax2.set_title("Acc",fontsize=16)
ax2.set_xlabel('Epoch', fontsize=16)
fig.savefig('./figure/ver10.png')

"""
# ver5
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver5_change_data_aug_ver_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver5_change_data_aug_ver_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver5_change_data_aug_ver_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver5_change_data_aug_ver_SGD_1e-3_epoch_100/test_acc.npy')

# ver6-2
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver6-2_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver6-2_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver6-2_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver6-2_SGD_1e-3_epoch_100/test_acc.npy')

# ver7
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver7_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver7_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver7_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver7_SGD_1e-3_epoch_100/test_acc.npy')

# ver8
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver8_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver8_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver8_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver8_SGD_1e-3_epoch_100/test_acc.npy')

# ver9
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver9_SGD_1e-3_epoch_100/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver9_SGD_1e-3_epoch_100/test_loss.npy')
train_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver9_SGD_1e-3_epoch_100/train_acc.npy')
test_acc = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver9_SGD_1e-3_epoch_100/test_acc.npy')

fig = plt.figure(figsize=(15,10), dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.arange(len(train_loss)), train_loss, label='train')
ax1.plot(np.arange(len(test_loss)), test_loss, label='test')
ax1.legend()
ax1.set_title("Loss",fontsize=16)
ax1.set_xlabel('Epoch', fontsize=16)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(len(train_acc)), train_acc, label='train')
ax2.plot(np.arange(len(test_acc)), test_acc, label='test')
ax2.legend()
ax2.set_title("Acc",fontsize=16)
ax2.set_xlabel('Epoch', fontsize=16)
fig.savefig('./figure/ver6-2.png')
"""

"""
# ver3
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver3_SGD_1e-3_epoch_200/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver3_SGD_1e-3_epoch_200/test_loss.npy')

# ver4
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver4_baseline_224_SGD_1e-3_epoch_200/train_loss.npy')
test_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver4_baseline_224_SGD_1e-3_epoch_200/test_loss.npy')

fig = plt.figure(figsize=(15,10), dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.arange(len(train_loss)), train_loss, label='train')
ax1.plot(np.arange(len(test_loss)), test_loss, label='test')
ax1.legend()
ax1.set_title("Loss",fontsize=16)
ax1.set_xlabel('Epoch', fontsize=16)
fig.savefig('./figure/ver5.png')
"""

"""
# ver1
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver1_Adam_1e-3_epoch_200/train_loss.npy')

# ver2
train_loss = np.load('/home/users/s19013225/workspace/2022.Class.DataAnalysisAndPratice2/checkpoint/ver2_SGD_1e-2_epoch_200/train_loss.npy')

fig = plt.figure(figsize=(15,10), dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.arange(len(train_loss)), train_loss, label='train')
ax1.legend()
ax1.set_title("Loss",fontsize=16)
ax1.set_xlabel('Epoch', fontsize=16)
fig.savefig('./figure/ver2.png')
"""