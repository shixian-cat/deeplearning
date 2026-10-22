from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import numpy as np

x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]
X=torch.tensor(x).float()
Y=torch.tensor(y).float()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)



class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x.clone().detach()
        self.y = y.clone().detach()
    def __len__(self):
        return len(self.x)
    def __getitem__(self,ix):
        return self.x[ix],self.y[ix]

ds=MyDataset(X,Y)
dl=DataLoader(ds,batch_size=2,shuffle=True)

class Myneuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer=nn.Linear(2,8)
        self.hidden_layer_to_activation=nn.ReLU()
        self.hidden_to_output_layer=nn.Linear(8,1)
    def forward(self,x):
        x=self.input_to_hidden_layer(x)
        x=self.hidden_layer_to_activation(x)
        x=self.hidden_to_output_layer(x)
        return x
mynet=Myneuralnet().to(device)

def my_mean_sequared_error(_y,y):
    loss=(_y-y)**2
    loss=loss.mean()
    return loss

loss_func=nn.MSELoss()
from torch.optim import SGD
opt=SGD(mynet.parameters(),lr=0.0001)

