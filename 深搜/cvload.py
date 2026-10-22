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
fmnist_trn=datasets.FashionMNIST(data_folder,download=True,train=True)
fmnist_val=datasets.FashionMNIST(data_folder,download=True,train=False)

tr_images=fmnist_trn.data
tr_targets=fmnist_trn.targets
val_image=fmnist_val.data
val_target=fmnist_val.targets

class FMNISTDataset(Dataset):
    def __init__(self,x,y):
        x=x.float()/255
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
    val=FMNISTDataset(val_image,val_target)
    val_dl=DataLoader(val,batch_size=len(val_image),shuffle=True)
    return trn_dl,val_dl

from torch.optim import SGD,Adam
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

trn_dl,val_dl=get_data()
model,loss_fn,optimizer=get_model()
state_dict=torch.load('E:/denghc/deeplearning/mymodel/test/model.pth')
model.load_state_dict(state_dict)
model.to(device)
trn_losses,trn_accuracies=[],[]
val_losses,val_accuracies=[],[]

for ix,batch in enumerate(iter(val_dl)):
    x,y=batch
    is_correct=accuracy(x,y,model)
    val_accuracies.extend(is_correct)
val_accuracy=np.mean(val_accuracies)
print(val_accuracy)



# for epoch in range(20):
#     print(epoch)
#     trn_epoch_losses,trn_epoch_accuracies=[],[]
#     val_epoch_losses,val_epoch_accuracies=[],[]

#     for ix,batch in enumerate(iter(trn_dl)):
#         x,y=batch
#         batch_loss=train_batch(x,y,model,optimizer,loss_fn)
#         trn_epoch_losses.append(batch_loss)
#     for ix,batch in enumerate(iter(trn_dl)):
#         x,y=batch
#         is_correct=accuracy(x,y,model)
#         trn_epoch_accuracies.extend(is_correct)
#     trn_epoch_loss=np.array(trn_epoch_losses).mean()
#     trn_epoch_accuracy=np.mean(trn_epoch_accuracies)
#     trn_losses.append(trn_epoch_loss)
#     trn_accuracies.append(trn_epoch_accuracies)
#     for ix,batch in enumerate(iter(val_dl)):
#         x,y=batch
#         is_correct=accuracy(x,y,model)
#         val_epoch_accuracies.extend(is_correct)
#         batch_loss=val_loss(x,y,model,loss_fn)
#         val_epoch_losses.append(batch_loss)

#     val_epoch_accuracy=np.mean(val_epoch_accuracies)
#     val_accuracies.append(val_epoch_accuracy)
#     val_epoch_loss=np.mean(val_epoch_losses)
#     val_losses.append(val_epoch_loss)


# save_path='E:/denghc/deeplearning/mymodel/test/model.pth'
# torch.save(model.state_dict(),save_path)

# epochs=np.arange(20)+1
# import matplotlib.ticker as mtick
# import matplotlib.ticker as mticker
# plt.subplot(121)                                      # 1行2列的第1个位置
# plt.plot(epochs, trn_losses, 'bo', label='Training loss')
# plt.plot(epochs, val_losses, 'r', label='Validation loss')

# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Loss value over increasing epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')  
# plt.grid('off') 
# plt.legend()        

# plt.subplot(122)                                      # 1行2列的第2个位置
# plt.title('Accuracy value over increasing epochs')    # 标题
# plt.plot(epochs, trn_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# from matplotlib.ticker import PercentFormatter
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.legend()          
# plt.grid('off')                        # 显示图例
# plt.show()

     