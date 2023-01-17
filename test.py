import os
import cv2
import numpy as np
import torch
from net import *
from dataset import *
from PIL import Image
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
net = unet(3,1).to(device)
weight = "unetx.pth"
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight, map_location="cpu"))
    print("successfully")
else:
    print("no loading")

input = input("please input image path:")
imgs = glob.glob(os.path.join(input,"image/*.jpg"))
print(imgs)
for img in imgs:
    image = np.array(Image.open(img).convert("RGB"))
    image = transforms(image).to(device)
    image = torch.unsqueeze(image, dim=0)
    net.eval()
    pred = net(image)
    pred = torch.squeeze(pred, dim=0)
    pred = torch.sigmoid(pred)
    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    #pred = pred.permute((1,2,0)).detach().numpy()
    pred = np.array(pred.data.cpu()[0])
    #print(pred.shape)
    save_path = img.replace("image","pred")
    cv2.imwrite(save_path,pred)

