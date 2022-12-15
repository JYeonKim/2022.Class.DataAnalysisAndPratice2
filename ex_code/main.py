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

from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
import gc

from tqdm import tqdm

def train(image_size, batch_size, num_workers, optimizer_name, learning_rate, nb_epoch):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    train_dir = "/home/users/s19013225/workspace" + "/data/reorganized" 
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts your input image to PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # dataset, dataloader 만들기
    print(">> make dataloader")
    all_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_dataset, test_dataset = data.random_split(all_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model 
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model.fc = torch.nn.Linear(1024, 7)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    model = model.to(device)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-3)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3, momentum=0.8)
    loss = torch.nn.CrossEntropyLoss()

    print(">> TRAIN Start")

    best_acc = 0

    log_train_loss = []
    log_test_loss = []
    
    for i in range(nb_epoch+1):
        model.train()
        pbar = tqdm(train_dataloader)

        train_pred = list()
        train_gt = list()
        batch_loss = list()

        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            
            try:
                hypothesis = model(x)
            except:
                import pdb; pdb.set_trace()
            
            cost = loss(hypothesis, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            batch_loss.append(cost.item())
            train_pred.extend(torch.argmax(hypothesis, dim=1).cpu().numpy())
            train_gt.extend(y.cpu().numpy())
            pbar.set_postfix({'epoch' : i, 'b_train_loss' : np.mean(batch_loss), 'b_train_acc' : accuracy_score(train_gt, train_pred)})

        # train loss 기록
        log_train_loss.append(np.mean(batch_loss))
        np.save(os.path.join(args.save_path, 'train_loss.npy'), np.array(log_train_loss))

        print("\n")
        # test 실행
        print(">>> Epoch : ", i)

        test_pred = list()
        test_gt = list()
        test_batch_loss = list()

        model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader)
            for (x, y) in test_pbar:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                cost = loss(pred, y)

                test_batch_loss.append(cost.item())
                test_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
                test_gt.extend(y.cpu().numpy())
                test_pbar.set_postfix({'epoch': i, 'b_test_acc' : accuracy_score(test_gt, test_pred)})
        
        # test loss 기록
        log_test_loss.append(np.mean(test_batch_loss))
        np.save(os.path.join(args.save_path, 'test_loss.npy'), np.array(log_test_loss))

        print("\n")

        message = 'Epoch {:3d}: accuracy {:5.3f}\n'.format(i, accuracy_score(test_gt, test_pred))
        
        if best_acc < accuracy_score(test_gt, test_pred):
            best_acc = accuracy_score(test_gt, test_pred)
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_acc.pt'))
            f = open(os.path.join(args.save_path, 'best_accuracy.txt'), "a")
            f.write(message)
            f.close()

        f = open(os.path.join(args.save_path, 'accuracy.txt'), "a")
        f.write(message)
        f.close()
        torch.save(model.state_dict() , os.path.join(args.save_path, str(i)+'.pt'))


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

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
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # image_size = 600
    image_size = 224
    batch_size = args.batch_size
    num_workers = args.num_workers
    optimizer_name = "SGD"
    learning_rate = 1e-3
    nb_epoch = 200

    train(image_size, batch_size, num_workers, optimizer_name, learning_rate, nb_epoch)

"""
실험 로그

OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/Adam_1e-3_epoch_200' --num_workers 30
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/SGD_1e-2_epoch_200' --num_workers 30
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/baseline_SGD_1e-3_epoch_200' --num_workers 30

OMP_NUM_THREADS=32 python main.py --gpu '0' --batch_size 16 --save_path './checkpoint/baseline_224_SGD_1e-3_epoch_200' --num_workers 30
"""