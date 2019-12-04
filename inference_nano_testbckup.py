from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNetV2
from utils import progress_bar
from datetime import datetime
import cv2

from os import listdir
from PIL import Image
import re
from datetime import datetime
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNetV2


##Inference on a single trained and saved model. The model is MobileNetV2
##This code is for the final dataset with 12 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
for params in parent_model.parameters():
     params.requires_grad=True

    
#fine-tuning with 2 fc layers on the model
parent_model.classifier[1] = nn.Sequential(nn.Linear(in_features=parent_model.classifier[1].in_features, out_features=512), nn.ReLU(),nn.Linear(in_features=512, out_features=12), nn.Softmax(dim=1))

print(parent_model.classifier)


#Loading saved models
parent_model.load_state_dict(torch.load('Mob_classifier_Shuufle.pth'),strict = False)
parent_model.to(device)
parent_model.eval()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
par_pred_p = []

sttpred = []

p_acc = 0
par_prob_ap = []

i = 0
avg_Time = 0


for image_file_name in sorted(listdir('./whole-n/')):
    #Put all images under a folder called "whole-n". 
    if image_file_name.endswith(".jpg"):
        imt = Image.open('./whole-n/'+image_file_name)
        tstart = datetime.now()
        #imt = np.array(imt).astype('float32')/255
        imt = transform_test(imt)
        feature_row = np.reshape(imt, (1,3,224,224))
        x = torch.tensor(feature_row,device=device)
        #tstart = datetime.now()
       
        par_prob = parent_model(x)
        p_cpu = par_prob.cpu().detach().numpy()
        #tend = datetime.now()
        par_pred = np.argmax(p_cpu, axis=1)
        tend = datetime.now()
        delta = tend - tstart
        avg_Time += (delta.total_seconds())*1000
        imfn = re.sub("\D", "", image_file_name)
        imnn = int(imfn)
        ##Images shoule be named in numerical way and in the right order of classes (Folders).
        if imnn <= 180:
            m_label = 0
        elif imnn in range(181,847):
            m_label = 4
        elif imnn in range(847,1029):
            m_label = 5
        elif imnn in range(1029,1128):
            m_label = 6
        elif imnn in range(1128,1468):
            m_label = 7
        elif imnn in range(1468,1768):
            m_label = 8
        elif imnn in range(1768,2107):
            m_label = 9
        elif imnn in range(2107,2206):
            m_label = 10
        elif imnn in range(2206,2609):
            m_label = 11
        elif imnn in range(2609,2729):
            m_label = 1
        elif imnn in range(2729,3340):
            m_label = 2
        else:
            m_label = 3
        if m_label == par_pred:
            p_acc_n=1
        else:
            p_acc_n=0
        p_acc += p_acc_n
        i += 1
        #print(i)


per_img_time = avg_Time/i


print('Per image running time is = ', per_img_time)

print('p_acc is', p_acc)
p_accuracy = 100.0*p_acc/i

##Accuracy of the model during the inference
print('p_accuracy is', p_accuracy)
