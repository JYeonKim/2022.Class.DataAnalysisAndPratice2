import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# To normalize the dataset, calculate the mean and std
print(">> check image mean, std")
train_meanR = 0.7640913724899292
train_meanG = 0.5459975600242615
train_meanB = 0.5704405903816223
train_stdR = 0.08975193649530411
train_stdG = 0.11854125559329987
train_stdB = 0.13313929736614227

print(">> Train")
print("meanR : {}, meanG : {}, meanB : {}".format(train_meanR, train_meanG, train_meanB))
print("stdR : {}, stdG : {}, stdB : {}".format(train_stdR, train_stdG, train_stdB))

def pil_to_tensor(pil_image):
    # PIL: [width, height]
    # -> NumPy: [width, height, channel]
    # -> Tensor: [channel, width, height]
    return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)

def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)

def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()


## 여기를 바꿔가며 확인
transform  = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=1),
    # transforms.RandomVerticalFlip(p=1),
    transforms.RandomPerspective(p=1),
    # transforms.RandomAffine(30),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
])
#######

img_path = "/home/users/s19013225/workspace/data/reorganized/akiec/ISIC_0024418.jpg"

plt.figure(figsize=(10, 10), dpi=200)
pil_image = PIL.Image.open(img_path)
applied_image = transform(pil_image)
plt.subplot(1, 1, 1)
plt.imshow(tensor_to_pltimg(applied_image))
plt.axis('off')
plt.show()
plt.savefig("./figure_data_aug_ver1/RandomPerspective.png")