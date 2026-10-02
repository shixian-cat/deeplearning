import os
import time
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

review_list = label_list = []
for label in ['pos', 'neg']:
    for fname in tqdm(os.listdir(f'./aclImdb/train/{label}/')):
            if 'txt' not in fname:continue
            
            with open(os.path.join(f'./aclImdb/train/{label}/', fname), encoding="utf8") as f:
                review_list += [f.read()]
                label_list += [label]
                
print ('Number of reviews :', len(review_list))