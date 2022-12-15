# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
# from torchsummary import summary
import time
import copy

class GoogleNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000, init_weights=True, dropout=0.2, dropout_aux=0.7):
        super(GoogleNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # # conv_block takes in_channels, out_channels, kernel_size, stride, padding
        # # Inception block takes out1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool

        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes, dropout_aux)
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout, True)
        self.fc = nn.Linear(1024, num_classes)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.maxpool2(x)

            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.maxpool3(x)
            x = self.inception4a(x)

            if self.aux_logits and self.training:
                aux1 = self.aux1(x)

            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)

            if self.aux_logits and self.training:
                aux2 = self.aux2(x)

            x = self.inception4e(x)
            x = self.maxpool4(x)
            x = self.inception5a(x)
            x = self.inception5b(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            aux3 = self.fc(x)

            if self.aux_logits and self.training:
                return aux3, aux1, aux2
            else:
                return aux3

    # define weight initialization function
    def _initialize_weights(self):

        # 먼저 모든 layer에 대해서 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # pretrained model의 weight, bias로 초기화
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # len(param_names) # 364
        # len(pretrained_param_names) # 344

        for i, name in enumerate(param_names[:-20-2]):
            state_dict[name] = pretrained_state_dict[pretrained_param_names[i]]
        self.load_state_dict(state_dict)

        print(">> Weight Init Finished <<")

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어준다
        
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]
        out = torch.cat(out, 1)
        return out

# auxiliary classifier의 loss는 0.3이 곱해지고, 최종 loss에 추가한다
# 정규화 효과 있음
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(InceptionAux, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size =3, stride=1, padding=0)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(512, 1024)
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout, True)

    def forward(self,x):
        with torch.autograd.set_detect_anomaly(True):
            out = self.avgpool(x)
            out = self.conv(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)

            return out