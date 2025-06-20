from torchinfo import summary
from thop import profile

import torch
import torch.nn as nn
import torch.optim as optim
from timm.models import vision_transformer as vits
from timm.data import ImageDataset
from torchvision import transforms
import argparse
from my_utils import read_data, evaluate, my_train_one_epoch, read_split_data, read_split_ADNI
from my_dataset import MyDataSet
from RVFL import RVFL
from torch.utils.tensorboard import SummaryWriter
import os
import time

import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from model.Myformer import Myformer

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings
import utils
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path


import warnings
from timm.scheduler.cosine_lr import CosineLRScheduler

from RVFL import RVFL
import torch.nn as nn
from timm.models import vision_transformer as vits

warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    
    device = torch.device(args.device1 if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device2 if torch.cuda.is_available() else "cpu")


    model = Myformer(num_class=args.num_classes,device1=args.device1, device2=args.device2)
    input = torch.rand((1, 3,288,288, 96)).cuda(args.device1)
     # 计算参数量
    model_summary = summary(model, input_size=(1, 3,288,288, 96))

    # 计算FLOPs
    flops, params = profile(model, inputs=(input, ), verbose=False)

    print(f"Total parameters: {model_summary.total_params/1e6:.4f} M")
    print(f"Total FLOPs: {flops/1e9:.4f} G")




    # S-AD-EMCI   acc 92.31     base 89.74    flat 87.18   cas 87.18
    # S-AD-LMCI   acc 90.32     base 87.10    flat 90.32   cas 93.55
    # S-AD-CN     acc 97.50     base 92.50    flat 92.50   cas 95.00
    # S-EMCI-LMCI acc 81.82     base 75.00    flat 75.00   cas 79.55
    # S-EMCI-CN   acc 84.91     base 73.58    flat 81.13   cas 79.24
    # S-LMCI-CN   acc 84.44     base 75.56    flat 84.44   cas 82.22

    # 1-AD-MCI   acc 92.31 17   base 87.69    flat 90.77   cas 86.15
    # 1-AD-CN    acc 92.68      base 85.37    flat 90.24   cas 92.68
    # 1-MCI-CN   acc 88.89 19   base 83.33    flat 87.50   cas 84.72

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00001) # bs=3 1-0.5 bs=1 0.3-0.1 


  
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data_path', type=str,
                        default="/mnt/data/lsy/ADNI-3d/ADNI-3D-4CLS")
      
    # 预训练权重路径，如果不想载入就设置为空字符 './swin_tiny_patch4_window7_224.pth'
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device1', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--device2', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_weight_path', default=r'/mnt/data/lsy/SAT/my_SAT_Nano3.pth', help='device id (i.e. 0 or 0,1 or cpu)')
    
    opt = parser.parse_args()

    main(opt)


        # original FLOPs 9.7979
    # linear
    # S AD-MCI 92.77 72900.57 [197, 54, 52, 52, 52, 51, 51, 51, 51, 51, 51, 71], [197, 52, 52, 52, 52, 51, 51, 51, 51, 51, 51, 71]
    # S AD-CN  87.66 13149.13 [197, 55, 53, 53, 53, 53, 53, 53, 53, 53, 53, 71], [197, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 71]
    # S CN-MCI 88.19 41409.26 [197, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # 300
    # A-c AD-MCI 88.95 13115.99 [197, 54, 52, 52, 51, 51, 51, 51, 51, 51, 51, 71], [197, 52, 52, 52, 51, 51, 51, 51, 51, 51, 51, 71]
    # A-c AD-CN 93.03 9352.05 [197, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A-c CN-MCI 90.03 16732.01 [197, 57, 52, 52, 52, 52, 51, 51, 51, 51, 51, 71], [197, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51, 71]
    
    # full token
    # [197, 101, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [197, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    
    # S AD-MCI 92.17 7309.00 [197, 54, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # S AD-CN  91.77 4889.74 [197, 55, 53, 53, 53, 53, 53, 53, 53, 53, 53, 71], [197, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 71]
    # S CN-MCI 93.99 11511.73 [197, 56, 54, 54, 54, 54, 54, 54, 54, 54, 54, 72], [197, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 72]
    
    # 300
    # A-c AD-MCI 90.02 12961.21 [197, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A-c AD-CN  90.05 14834.94 [197, 54, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A-c AD-CN2  93.41 12428.60 [197, 55, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A-c CN-MCI 90.95 17389.20 [197, 56, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71], [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    


    
    # A AD-MCI 91.52 15815.65 [197, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71] [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A AD-CN  87.54 10893.48 [197, 54, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71] [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A CN-MCI 

    # A AD-MCI 90.92 9848.85 [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71] [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A AD-CN  86.98 6526.10 [197, 54, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71] [197, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 71]
    # A CN-MCI 