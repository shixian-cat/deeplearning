from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
  
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn

X=torch.tensor([[1,2],[3,4],[5,6],[7,8]]).float()
Y=torch.tensor([[3],[7],[11],[15]]).float()

device='cuda' if torch.cuda.is_available() else 'cpu'
X=X.to(device)
Y=Y.to(device)
data_folder='E:/denghc/deeplearning/data'
fmnist=datasets.FashionMNIST(data_folder,download=True,train=True)

tr_images=fmnist.data
tr_targets=fmnist.targets


class FMNISTDataset(Dataset):
    def __init__(self,x,y):
        x=x.float()
        x=x.view(-1,28*28)
        self.x,self.y=x,y
    def __getitem__(self,ix):
        x,y=self.x[ix],self.y[ix]
        return x.to(device),y.to(device)
    def __len__(self):
        return len(self.x)
    
def get_data():
    train=FMNISTDataset(tr_images,tr_targets)
    trn_dl=DataLoader(train,batch_size=32,shuffle=True)
    return trn_dl

from torch.optim import SGD
def get_model():
    model=nn.Sequential(
        nn.Linear(28*28,1000),
        nn.ReLU(),
        nn.Linear(1000,10)
    ).to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(),lr=1e-2)
    return model,loss_fn,optimizer
def train_batch(x,y,model,optimizer,loss_fn):
    model.train()
    prediction=model(x)
    batch_loss=loss_fn(prediction,y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()
@torch.no_grad()
def accuracy(x,y,model):
    model.eval()
    prediction=model(x)
    max_values,argmaxes,=prediction.max(-1)
    is_correct=argmaxes==y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x,y,model,loss_fn):
    prediction=model(x)
    val_loss=loss_fn(prediction,y)
    return val_loss.item()

trn_dl=get_data()
model,loss_fn,optimizer=get_model()
losses,accuracies=[],[]
for epoch in range(10):
    print(epoch)
    epoch_losses,epoch_accuracies=[],[]
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        batch_loss=train_batch(x,y,model,optimizer,loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss=np.array(epoch_losses).mean()
    for ix,batch in enumerate(iter(trn_dl)):
       x,y=batch
       is_correct=accuracy(x,y,model)
       epoch_accuracies.extend(is_correct)
    epoch_accuracy=np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracies)

epochs=np.arange(10)+1
plt.figure(figsize=(20,5))
plt.subplot(121)                                      # 1行2列的第1个位置
plt.title('Loss value over increasing epochs')        # 标题
plt.plot(epochs, losses, label='Training Loss')       # 画折线图
plt.legend()        
plt.subplot(122)                                      # 1行2列的第2个位置
plt.title('Accuracy value over increasing epochs')    # 标题
plt.plot(epochs, accuracies, label='Training Accuracy') # 画折线图
plt.gca().set_yticklabels(['{:0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()                                  # 显示图例
plt.show()

  

# R,C=len(tr_targets.unique()),10
# fig,ax=plt.subplots(R,C,figsize=(10,10))
# for label_class,plot_row in enumerate(ax):
#     label_x_rows=np.where(tr_targets==label_class)[0]
#     for plot_cell in plot_row:
#         plot_cell.grid(False)
#         plot_cell.axis('off')
#         ix=np.random.choice(label_x_rows)
#         x,y=tr_images[ix],tr_targets[ix]
#         plot_cell.imshow(x,cmap='gray')
# plt.show()