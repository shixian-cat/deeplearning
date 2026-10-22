import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np, cvision2
import matplotlib.pyplot as plt
from glob import glob
from imgaug import augmenters as iaa

tfm=iaa.Sewurntial(iaa.Resize(28))

class X0(Dataset):
    def __init__(self,folder):
        self.files=glob(folder)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        f=self.files[index]
        im=tfm.augment_image(cvision2.imread(f)[:,:,0])
        im=im[None]
        c1=f.split('/')[-1].split('@')[0]=='x'
        return torch.tensor(1-im/255).to(device).float(),torch.tensor(c1).float.to(device)

data=X0('data/images/*')
R,C=7,7
fig,ax=plt.subplots(R,C,figsize=(5,5))
for label,plot_row in enumerate(ax):
    for plot_cell in plot_row:
        plot_cell.grid(False);plot_cell.axis('off')
        ix=np.random.choice(1000)
        im,label=data[ix]
        plot_cell.imshow(im[0].cpu(),cmap='gray')
plt.tight_layout()
plt.show()

from torch.optim import SGD,Adam
def get_model():
    model=nn.Sequential(
        nn.Conv2d(1,64,kernek_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64,128,kernek_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200,256),
        nn.ReLU(),
        nn.Linear(256,1),
        nn.Sigmoid()
    ).to(device)
    loss_fn=nn.BCELoss()
    optimizer=Adam(model.parameters(),lr=1e-3)
    return model,loss_fn,optimizer

def train_batch(x,y,model,opt,loss_fn):
    prediction=model(x)
    batch_loss=loss_fn(prediction,y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

def accuracy(x,y,model):
    with torch.no_grad():
        prediction=model(x)
    max_values,argmaxes=prediction.max(-1)
    is_correct=argmaxes==y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x,y,model,loss_fn):
    prediction=model(x)
    val_loss=loss_fn(prediction,y)
    return val_loss.item()

trn_dl=DataLoader(data,batch_size=32,drop_last=True)
model,loss_fn,optimizer=get_model()

for epoch in range(10):
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        batch_loss=train_batch(x,y,model,optimizer,loss_fn)

im,c=trn_dl.dataset[2]
plt.imshow(im[0].cpu())
plt.show()

# print(list(model.children()))
# [Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1)), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), ReLU(), Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), ReLU(), Flatten(start_dim=1, end_dim=-1), Linear(in_features=3200, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=1, bias=True), Sigmoid()]

first_layer=nn.Sequential(*list(model.children())[:1])
intermediate_output=first_layer(im[None])[0].detach()

n=8
fig,ax=plt.subplot(n,n,figsize=(10,10))
for ix,axis in enumerate(ax.flat):
   axis.set_title(f'Filter {ix}')
   axis.imshow(intermediate_output[ix].cpu(),cmap='gray')
plt.tight_layout()
plt.show()

x,y=next()