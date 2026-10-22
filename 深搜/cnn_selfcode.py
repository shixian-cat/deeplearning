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
    def __init__(self):
        super(AutoEncoder,self).__init__()
        # self.latent_dim=latent_dim
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,3,stride=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=1)
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,5,stride=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,2,stride=2,padding=1),
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

model=AutoEncoder()
# model=model.to(device)
# criterion=nn.MSELoss()
# optimizer=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    
# num_epochs=20
# trn_loss=[]
# val_loss=[]
# for epoch in range(num_epochs):
#     N=len(trn_dl)
#     trn_loss_epoch=[]
#     val_loss_epoch=[]
#     for i,(x,_) in enumerate(trn_dl):
#         x = x.to(device)
#         # x = x.view(x.size(0), -1)
#         loss = train(x, model, criterion, optimizer)
#         pos=(epoch+(i+1)/N)
#         trn_loss_epoch.append(loss.item())
#         print(f'\rEpoch {epoch+1}/{num_epochs} | Batch {i+1}/{N} | Loss: {loss.item():.4f}',end='')
#     trn_loss.append(np.mean(trn_loss_epoch))

#     N=len(val_dl)
#     for i,(x,_) in enumerate(val_dl):
#         x = x.to(device)
#         # x = x.view(x.size(0), -1)
#         loss = validate(x, model, criterion)
#         pos=(epoch+(i+1)/N)
#         print(f'\rEpoch {epoch+1}/{num_epochs} | Batch {i+1}/{N} | Loss: {loss.item():.4f}',end='')
#         val_loss_epoch.append(loss.item())
#     val_loss.append(np.mean(val_loss_epoch))

# epochs=np.arange(num_epochs)+1
# plt.plot(epochs,trn_loss,label='train')
# plt.plot(epochs,val_loss,label='val')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Train and Validation Loss')
# plt.legend()
# plt.grid('off')
# plt.show()

# save_path='E:/denghc/deeplearning/mymodel/cnn_selfcode/model.pth'
# torch.save(model.state_dict(),save_path)

# for _ in range(5):
#     ix=np.random.randint(len(val_ds))
#     im,_=val_ds[ix]
#     _im=model(im[None])[0]
#     plt.subplot(121)
#     plt.imshow(im[0].detach().cpu(),cmap='gray')
#     plt.title('Original Image')
#     plt.subplot(122)
#     plt.imshow(_im[0].detach().cpu(),cmap='gray')
#     plt.title('Reconstructed Image')
#     plt.show()

state_dict=torch.load('E:/denghc/deeplearning/mymodel/cnn_selfcode/model.pth')
model.load_state_dict(state_dict)
model.to(device)

latent_vectors=[]
classes=[]

for im,clss in val_dl:
    latent_vectors.append(model.encoder(im).view(len(im), -1))
    classes.append(clss)

latent_vectors=torch.cat(latent_vectors).cpu().detach().numpy()

from sklearn.manifold import TSNE
tsne=TSNE(2)

clustered=tsne.fit_transform(latent_vectors)

fig=plt.figure(figsize=(12,10))
cmap=plt.get_cmap('Spectral',10)
plt.scatter(*zip(*clustered),c=classes,cmap=cmap)
plt.colorbar(drawedges=True)
plt.show()







