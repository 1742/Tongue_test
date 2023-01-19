import torch
import glob
import torchvision
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

transformer = transforms.Compose([
    transforms.ToTensor()
])


class dataset(Dataset):
    def __init__(self,data_path,transform=None):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.transform = transform

    def __getitem__(self,index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("image","label")
        image = Image.open(image_path).convert("RGB")
        label = np.array(Image.open(label_path).convert("L"),dtype=np.float32)
        label[label>0] = 1.0
        if self.transform is not None:
            image = transformer(image)
            label = transformer(label)
        return image ,label

    def __len__(self):
        return len(self.imgs_path)