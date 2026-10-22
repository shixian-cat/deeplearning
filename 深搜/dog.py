import torch
from torchvision import transforms,models,datasets
device='cuda' 
model=models
import torch.nn as nn

import matplotlib.pyplot as plt
import cv2, glob, numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import DataLoader, Dataset

trn_data_dir='E:/denghc/deeplearning/data/cat/training_set/training_set'
test_data_dir='E:/denghc/deeplearning/data/cat/test_set/test_set'

class CatsDogs(Dataset):
    def __init__(self,folder):
        cats=glob(folder+'/cats/*.jpg')
        dogs=glob(folder+'/dogs/*.jpg')
        self.fpaths=cats[:500]+dogs[:500]
        self.normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        from random import shuffle,seed;seed(10);shuffle(self.fpaths)
        self.targets=[fpath.split('/')[-1].startswith('cat') for fpath in self.fpaths]
    def __len__(self):
        return len(self.fpaths)
    def __getitem__(self, index):
        fpath=self.fpaths[index]
        im=(cv2.imread(fpath)[:,:,::-1])
        im=cv2.resize(im,(224,224))
        im=torch.tensor(im).permute(2,0,1).float()/255
        im=self.normalize(im)
        target=torch.tensor(self.targets[index]).float()
        return im.float(),torch.tensor([target]).float()



def get_model():
    model=models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    model.avgpool=nn.AdaptiveAvgPool2d((1,1))
    model.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
    )
    model=model.to(device)
    loss_fn=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    return model,optimizer,loss_fn


def train_batch(x,y,model,optimizer,loss_fn):
    model.train()
    prediction=model(x)
    batch_loss=loss_fn(prediction,y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

def accuracy(x,y,model):
    with torch.no_grad():
        prediction=model(x)
    max_values,argmaxes=prediction.max(-1)
    is_correct=argmaxes==y
    return is_correct.cpu().numpy().tolist()

def get_data():
    trn_data=CatsDogs(trn_data_dir)
    val_data=CatsDogs(test_data_dir)
    trn_dl=DataLoader(trn_data,batch_size=32,shuffle=True)
    val_dl=DataLoader(val_data,batch_size=32,shuffle=False)
    return trn_dl,val_dl

@torch.no_grad()
def val_loss(x,y,model,loss_fn):
    model.eval()
    prediction=model(x)
    val_loss=loss_fn(prediction,y)
    return val_loss.item()

trn_dl,val_dl=get_data()
model,optimizer,loss_fn=get_model()

trn_loss,trn_acc=[],[]
val_acc,val_accs=[],[]
for epoch in range(10):
    print(f"epoch{epoch+1}/10")
    trn_epoch_loss,trn_epoch_accuracies=[],[]
    val_epoch_loss,val_epoch_accuracies=[],[]
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        x,y=x.to(device),y.to(device)
        batch_loss=train_batch(x,y,model,optimizer,loss_fn)
        trn_epoch_loss.append(batch_loss)
    trn_epoch_losses=np.mean(trn_epoch_loss)
    trn_loss.append(trn_epoch_losses)

    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        x,y=x.to(device),y.to(device)
        batch_acc=accuracy(x,y,model)
        trn_epoch_accuracies+=batch_acc
    trn_epoch_accuracy=np.mean(trn_epoch_accuracies)
    trn_acc.append(trn_epoch_accuracy)

    for ix,batch in enumerate(iter(val_dl)):
        x,y=batch
        x,y=x.to(device),y.to(device)
        batch_acc=accuracy(x,y,model)
        val_epoch_accuracies+=batch_acc
    val_epoch_accuracy=np.mean(val_epoch_accuracies)
    val_acc.append(val_epoch_accuracy)

epochs=np.arange(10)+1
import matplotlib.ticker as mtick
plt.figure(figsize=(16, 8))  # 宽度12英寸，高度
plt.plot(epochs,trn_acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'r',label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid('off')
plt.show()