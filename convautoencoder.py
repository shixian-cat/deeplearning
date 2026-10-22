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

img_tran=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.to(device))
])

trn_ds=MNIST(root='./data/MNIST', train=True, download=True, transform=img_tran)
val_ds=MNIST(root='./data/MNIST', train=False, download=True, transform=img_tran)

batch_size=256
trn_dl=DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl=DataLoader(val_ds, batch_size=batch_size, shuffle=False)

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=1),
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1), 
            nn.Tanh()
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


def trn(imput,model,cri,opt):
    model.train()
    opt.zero_grad()
    output=model(imput)
    loss=cri(output,imput)
    loss.backward()
    opt.step()
    return loss.item()

def val(imput,model,cri):
    model.eval()
    with torch.no_grad():
        output=model(imput)
        loss=cri(output,imput)
    return loss.item()

def get_model():
    model=ConvAutoEncoder()
    cri=nn.MSELoss()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    return model.to(device),cri,opt
model,cri,opt=get_model()

n_epochs=20
trn_losses,val_losses=[],[]
for epoch in range(n_epochs):
    trn_loss=0
    for i, data in enumerate(trn_dl):
        im,_=data
        im=im.to(device)
        loss=trn(im,model,cri,opt)
        trn_loss+=loss
    trn_loss/=len(trn_dl)
    trn_losses.append(trn_loss)

    val_loss=0
    for i, data in enumerate(val_dl):
        im,_=data
        im=im.to(device)
        loss=val(im,model,cri)
        val_loss+=loss
    val_loss/=len(val_dl)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}")

    
epochs = np.arange(n_epochs)+1
plt.plot(epochs, trn_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r-', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

model_path = './pth/conv_autoencoder.pth'
torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))

for _ in range(5):
    ix = np.random.randint(len(val_ds))
    im,_= val_ds[ix]
    im = model(im[None])[0]
    plt.subplot(121)
    plt.imshow(im[0].detach().cpu(), cmap='gray')
    plt.title('input')
    plt.subplot(122)
    plt.imshow(im[0].detach().cpu(), cmap='gray')
    plt.title('prediction')
    plt.show()