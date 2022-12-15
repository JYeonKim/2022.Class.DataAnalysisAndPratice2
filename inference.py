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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tqdm import tqdm

def inference(image_size, batch_size, num_workers, model_path):
    
    print(">> DATALOADER ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    train_dir = os.getcwd() + "/data/reorganized"
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts your input image to PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_dataset, test_dataset = data.random_split(all_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(">> Model")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model.fc = torch.nn.Linear(1024, 7)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # print(">>TRAIN PRED, GT")
    # model.eval()
    # pbar = tqdm(train_dataloader)

    # train_pred = list()
    # train_gt = list()
    # with torch.no_grad():
    #     for (x, y) in pbar:
    #         x = x.to(device)
    #         y = y.to(device)
            
    #         hypothesis = model(x)
    #         train_pred.extend(torch.argmax(hypothesis, dim=1).cpu().numpy())
    #         train_gt.extend(y.cpu().numpy())

    test_pred = list()
    test_gt = list()

    print("TEST PRED, GT")
    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(test_dataloader)
        for (x, y) in test_pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            test_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
            test_gt.extend(y.cpu().numpy())

    acc = accuracy_score(test_gt, test_pred)
    precision = precision_score(test_gt, test_pred, average='macro')
    recall = recall_score(test_gt, test_pred, average='macro')
    f1 = f1_score(test_gt, test_pred, average='macro')
    # auc = roc_auc_score(test_gt, test_pred, average='ovr')

    print(" ACC : {} \n Precision : {} \n Recall : {} \n F1 : {}\n".format(acc, precision, recall, f1))


if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='Evaluation_baseline', formatter_class=formatter)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    image_size = 600
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_path = './checkpoint/baseline_/best_acc.pt'

    inference(image_size, batch_size, num_workers, model_path)

"""
실험 로그

OMP_NUM_THREADS=1 python inference.py --gpu '1' --batch_size 16

"""