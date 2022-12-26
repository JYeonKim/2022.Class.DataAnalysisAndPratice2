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
import gc

from model import GoogleNet

from tqdm import tqdm

def loss_batch(loss, outputs, target, opt, loss_list, pred_list, gt_list):

    try:
        shape_size = np.shape(outputs)[0]
    except:
        shape_size = 3

    if shape_size == 3:
        output, aux1, aux2 = outputs

        output_loss = loss(output, target)
        aux1_loss = loss(aux1, target)
        aux2_loss = loss(aux2, target)

        plus_aux = aux1_loss.item() + aux2_loss.item()
        cost = output_loss + 0.3 * plus_aux
        hypothesis = output.detach().cpu()
    else:
        cost = loss(outputs, target)
        hypothesis = outputs.detach().cpu()

    if opt is not None:
        opt.zero_grad()
        cost.backward()
        opt.step()
    
    loss_list.append(cost.item())
    pred_list.extend(torch.argmax(hypothesis, dim=1).numpy())
    gt_list.extend(target.cpu().numpy())

    return loss_list, pred_list, gt_list

def train(image_size, batch_size, num_workers, optimizer_name, learning_rate, nb_epoch):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_dir을 파일 위치에 맞춰 변경함
    train_dir = "/home/users/s19013225/workspace/data/train_augmented"
    test_dir = "/home/users/s19013225/workspace/data/test"
    
    # dataset, dataloader
    print(">> make dataloader")
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms.ToTensor())
    
    # To normalize the dataset, calculate the mean and std
    print(">> check image mean, std")
    """
    # 이전에 코드를 실행시켜 확인하였음. (매번 실행할 때마다 굳이 확인할 필요 없기 때문에 주석 처리함)
    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in tqdm(train_dataset)]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in tqdm(train_dataset)]
    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])
    """
    train_meanR = 0.7640913724899292
    train_meanG = 0.5459975600242615
    train_meanB = 0.5704405903816223
    train_stdR = 0.08975193649530411
    train_stdG = 0.11854125559329987
    train_stdB = 0.13313929736614227

    print(">> Train")
    print("meanR : {}, meanG : {}, meanB : {}".format(train_meanR, train_meanG, train_meanB))
    print("stdR : {}, stdG : {}, stdB : {}".format(train_stdR, train_stdG, train_stdB))
    """
    >> Train
    meanR : 0.7640913724899292, meanG : 0.5459975600242615, meanB : 0.5704405903816223
    stdR : 0.08975193649530411, stdG : 0.11854125559329987, stdB : 0.13313929736614227
    """

    # define the image transformation
    train_transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomAffine(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    ])
    val_transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]), # test_dataloader의 mean, std를 이용하여 치팅이라고 판단하여 train으로 normalize 진행하였음.
    ])

    # apply transformation
    train_dataset.transform = train_transformation
    test_dataset.transform = val_transformation

    # create DataLoader
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model 정의
    model = GoogleNet(aux_logits=True, num_classes=7, init_weights=True)
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
    log_train_acc = []
    log_test_acc = []
    
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
                output = model(x)
            except:
                import pdb; pdb.set_trace()
            
            # loss 처리
            batch_loss, train_pred, train_gt  = loss_batch(loss, output, y, optimizer, batch_loss, train_pred, train_gt)

            pbar.set_postfix({'epoch' : i, 'b_train_loss' : np.mean(batch_loss), 'b_train_acc' : accuracy_score(train_gt, train_pred)})

        # train loss, acc 기록
        log_train_loss.append(np.mean(batch_loss))
        np.save(os.path.join(args.save_path, 'train_loss.npy'), np.array(log_train_loss))
        log_train_acc.append(accuracy_score(train_gt, train_pred))
        np.save(os.path.join(args.save_path, 'train_acc.npy'), np.array(log_train_acc))

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

                output = model(x)
                try:
                    shape_size = np.shape(output)[0]
                except:
                    shape_size = 3
                    
                # test는 aux loss 사용 x
                if shape_size == 3:
                    test_output, _, _ = output
                else:
                    test_output = output

                # loss 처리
                test_batch_loss, test_pred, test_gt  = loss_batch(loss, test_output, y, None, test_batch_loss, test_pred, test_gt)

                test_pbar.set_postfix({'epoch': i, 'b_test_acc' : accuracy_score(test_gt, test_pred)})
        
        # test loss, acc 기록
        log_test_loss.append(np.mean(test_batch_loss))
        np.save(os.path.join(args.save_path, 'test_loss.npy'), np.array(log_test_loss))
        log_test_acc.append(accuracy_score(test_gt, test_pred))
        np.save(os.path.join(args.save_path, 'test_acc.npy'), np.array(log_test_acc))

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

    image_size = 224
    batch_size = args.batch_size
    num_workers = args.num_workers
    optimizer_name = "SGD"
    learning_rate = 1e-3
    nb_epoch = 100

    train(image_size, batch_size, num_workers, optimizer_name, learning_rate, nb_epoch)

"""
실험 로그

OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/Adam_1e-3_epoch_200' --num_workers 30
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/SGD_1e-2_epoch_200' --num_workers 30
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/SGD_1e-3_epoch_200' --num_workers 30

# 224 ver4

# ver5
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/change_data_aug_ver_SGD_1e-3_epoch_100' --num_workers 30

# 연습 및 확인용
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/pratice' --num_workers 30

# pretrained 모델로 init + aux loss
OMP_NUM_THREADS=1 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/ver6-2_SGD_1e-3_epoch_100' --num_workers 30

# pretrained 모델로 init + aux loss x 버전
OMP_NUM_THREADS=32 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/ver7_SGD_1e-3_epoch_100' --num_workers 30

# pretrained 모델로 init + aux loss 0 + augmented data 0 버전
OMP_NUM_THREADS=32 python main.py --gpu '0' --batch_size 16 --save_path './checkpoint/ver8_SGD_1e-3_epoch_100' --num_workers 30

# pretrained 모델로 init + aux loss X + augmented data 0 버전
OMP_NUM_THREADS=32 python main.py --gpu '0' --batch_size 16 --save_path './checkpoint/ver9_SGD_1e-3_epoch_100' --num_workers 30

# pretrained 모델로 init + aux loss 0(test x, train 0) + augmented data 0 버전
OMP_NUM_THREADS=32 python main.py --gpu '1' --batch_size 16 --save_path './checkpoint/ver10_SGD_1e-3_epoch_100' --num_workers 30

# pretrained 모델로 init + aux loss 0(test x, train 0) + augmented data 0 버전 (수정한버전!!!!)
OMP_NUM_THREADS=32 python main.py --gpu '0' --batch_size 16 --save_path './checkpoint/ver11_SGD_1e-3_epoch_100' --num_workers 30
"""