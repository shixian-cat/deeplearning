from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)

x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,ix):
        return self.x[ix],self.y[ix]

ds=MyDataset(X,Y)
dl=DataLoader(ds,batch_size=2,shuffle=True)

model=nn.Sequential(
    nn.linear(2,8),
    nn.ReLU(),
    nn.linear(8,1)
).to(device)