from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

trn_ds=MNIST('data/MNIST',train=True,transform=img_transform,download=True)
val_ds=MNIST('data/MNIST',train=False,transform=img_transform,download=True)

batch_size = 256
trn_dl=DataLoader(trn_ds,batch_size=batch_size, shuffle=True)
val_dl=DataLoader(val_ds,batch_size=batch_size, shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim):
        super(AutoEncoder,self).__init__()
        self.latent_dim=latent_dim
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,latent_dim),
        )
        self.decoder=nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
           nn.Tanh()
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

def train(input,model,criterion,optimizer):
    model.train()
    optimizer.zero_grad()
    output=model(input)
    loss=criterion(output,input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate(input,model,criterion):
    model.eval()
    output=model(input)
    loss=criterion(output,input)
    return loss

model=AutoEncoder(3).to(device)
criterion=nn.MSELoss()
optimizer=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    
num_epochs=20
trn_loss=[]
val_loss=[]
for epoch in range(num_epochs):
    N=len(trn_dl)
    trn_loss_epoch=[]
    val_loss_epoch=[]
    for i,(x,_) in enumerate(trn_dl):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        loss = train(x, model, criterion, optimizer)
        pos=(epoch+(i+1)/N)
        trn_loss_epoch.append(loss.item())
        print(f'\rEpoch {epoch+1}/{num_epochs} | Batch {i+1}/{N} | Loss: {loss.item():.4f}',end='')
    trn_loss.append(np.mean(trn_loss_epoch))

    N=len(val_dl)
    for i,(x,_) in enumerate(val_dl):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        loss = validate(x, model, criterion)
        pos=(epoch+(i+1)/N)
        print(f'\rEpoch {epoch+1}/{num_epochs} | Batch {i+1}/{N} | Loss: {loss.item():.4f}',end='')
        val_loss_epoch.append(loss.item())
    val_loss.append(np.mean(val_loss_epoch))

epochs=np.arange(num_epochs)+1
plt.plot(epochs,trn_loss,label='train')
plt.plot(epochs,val_loss,label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid('off')
plt.show()










