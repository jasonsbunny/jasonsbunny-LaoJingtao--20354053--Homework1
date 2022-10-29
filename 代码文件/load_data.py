import os
import torchvision.transforms as transforms
import torch
import torchvision
import pandas as pd
import Data_Augmentation
import random
import numpy as np

#########################加载数据##############################
def read_data(is_train=True):
    """读取检测数据集中的图像和标签"""
    data_dir = 'detection/'
    csv_fname = os.path.join(data_dir, 'sysu_train' if is_train
                             else 'sysu_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        img=torchvision.io.read_image(
            os.path.join(data_dir, 'sysu_train' if is_train else
                         'sysu_val', 'images', f'{img_name}'))

        images.append(img)
        target = list(target)
        targets.append(target)

        x=random.random()
        if x>=0.5:
            box=target[1:]
            label=target[0]
            data_augmentation=Data_Augmentation.SSDAugmentation()
            img=img.numpy()
            img=img.transpose(1,2,0)
            img,boxs,labels = data_augmentation.__call__(img,box,label)
            img=transforms.ToTensor()(img)
            target0=[]
            target0.append(labels)
            for i in boxs:
                target0.append(i)
            target0 = np.array(target0).astype(dtype=int).tolist()
            images.append(img)
            targets.append(target0)
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的类（索引为0）
    return images, torch.tensor(targets).unsqueeze(1) / 256

class Dataset(torch.utils.data.Dataset):
    """一个用于加载检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data(batch_size):
    """加载检测数据集"""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True),
                                             batch_size, shuffle=True)
    #val_iter = torch.utils.data.DataLoader(Dataset(is_train=False),
    #                                       batch_size)
    return train_iter#, val_iter

