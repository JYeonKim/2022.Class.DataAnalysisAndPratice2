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


train_data = np.load('./checkpoint/SGD_1e-3_epoch_200/train_loss.npy')
test_data = np.load('./checkpoint/SGD_1e-3_epoch_200/test_loss.npy')

import pdb; pdb.set_trace()

plt.plot(np.arange(len(train_data)), train_data)
plt.plot(np.arange(len(test_data)), test_data)
plt.savefig("./checkpoint/SGD_1e-3_epoch_200/loss_fig.png")