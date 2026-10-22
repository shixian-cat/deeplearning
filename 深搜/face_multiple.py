import torch
import numpy as np, cv2, pandas as pd, time
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
device = 'cuda' 
trn_df=pd.read_csv('E:/denghc/deeplearning/data/Multi_Task_Learning/fairface-labels-train.csv')
val_df=pd.read_csv('E:/denghc/deeplearning/data/Multi_Task_Learning/fairface-labels-val.csv')

import os
BASE_DIR = r'E:\denghc\deeplearning\data\Multi_Task_Learning\fairface-img-margin025-trainval'

IMAGE_SIZE=224
class GenderAgeClass(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        f=self.df.iloc[idx].squeeze()
        img_path=os.path.join(BASE_DIR, f.file)
        gen=f.gender=='Female'
        age=f.age
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, age, gen

    def preprocess_image(self,im):
        im-=cv2.resize(im,(IMAGE_SIZE,IMAGE_SIZE))
        im=torch.tensor(im).permute(2,0,1).float()
        im=self.normalize(im/255)
        return im[None]

    def collate_fn(self, batch):
        'preprocesses images,ages and genders'
        images,ages,gender=[],[],[]
        for im,age,gen in batch:
            images.append(self.preprocess_image(im))
            ages.append(float(int(age)/80))
            gender.append(float(gen))
        ages,gender=torch.tensor(ages).float().to(device),torch.tensor(gender).float().to(device)
        return torch.cat(images,0).to(device),ages,gender

trn=GenderAgeClass(trn_df)
val=GenderAgeClass(val_df)

trn_loader=DataLoader(trn,batch_size=32,shuffle=True,drop_last=True,collate_fn=trn.collate_fn)
val_loader=DataLoader(val,batch_size=32,shuffle=True,drop_last=True,collate_fn=val.collate_fn)

def get_model():
    model=models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    model.avgpool=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    class GenderAgeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.intermediate=nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU()
            )
            self.age=nn.Sequential(
                nn.Linear(64,1),
                nn.Sigmoid()
            )
            self.gender=nn.Sequential(
                nn.Linear(64,1),
                nn.Sigmoid()
            )
        def forward(self,x):
            x=self.intermediate(x)
            age=self.age(x)
            gender=self.gender(x)
            return gender,age

    model.classifier=GenderAgeModel()
    gender_ctriterion=nn.BCELoss()
    age_criterion=nn.L1Loss()
    loss_fn=gender_ctriterion,age_criterion
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    return model.to(device),loss_fn,optimizer

model,loss_fn,optimizer=get_model()

def train_batch(data,model,optimizer,criteria):
    model.train()
    ims,age,gender=data
    optimizer.zero_grad()
    gender_pred,age_pred=model(ims)
    gender_criterion,age_criterion=criteria
    gender_loss=gender_criterion(gender_pred.squeeze(),gender)
    age_loss=age_criterion(age_pred.squeeze(),age)
    total_loss=gender_loss+age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss

def validate_batch(data,model,criteria):
    model.eval()
    ims,age,gender=data
    with torch.no_grad():
        gender_pred,age_pred=model(ims)
    gender_criterion,age_criterion=criteria
    gender_loss=gender_criterion(gender_pred.squeeze(),gender)
    age_loss=age_criterion(age_pred.squeeze(),age)
    total_loss=gender_loss+age_loss
    gender_pred=(gender_pred>0.5).squeeze()
    gender_acc=(gender_pred==gender).float().sum()
    age_mae=torch.abs(age_pred.squeeze()-age).float().sum()

    return total_loss,gender_acc,age_mae

val_gender_accs,val_age_maes,val_loss=[],[],[]
trn_loss=[]

n_epoch=10
best_test_loss=10000
start=time.time()
for epoch in range(n_epoch):
    _n=len(trn_loader)
    trn_epoch_loss,val_epoch_loss=0,0
    val_gender_acc,val_age_mae,ctr=0,0,0
    for ix,data in enumerate(trn_loader):
        loss=train_batch(data,model,optimizer,loss_fn)
        trn_epoch_loss+=loss.item()
        print(epoch,ix,_n,loss.item())
    for ix,data in enumerate(val_loader):
        loss,gender_acc,age_mae=validate_batch(data,model,loss_fn)
        val_epoch_loss+=loss.item()
        val_gender_acc+=gender_acc.item()
        val_age_mae+=age_mae.item()
        ctr+=len(data[0])
    trn_epoch_loss/=len(trn_loader)
    val_epoch_loss/=len(val_loader)
    val_gender_acc/=ctr
    val_age_mae/=ctr
    elapsed=time.time()-start
    best_test_loss=min(best_test_loss,val_epoch_loss)
    print(f"epoch{epoch+1}/{n_epoch} trn_loss:{trn_epoch_loss:.4f} val_loss:{val_epoch_loss:.4f} val_gender_acc:{val_gender_acc:.4f} val_age_mae:{val_age_mae:.4f}")
    info=f"best_test_loss:{best_test_loss:.4f} elapsed:{elapsed:.2f}s"
    print(info)                                                                                         
    val_gender_accs.append(val_gender_acc)
    val_age_maes.append(val_age_mae)
    val_loss.append(val_epoch_loss)
    trn_loss.append(trn_epoch_loss)

epochs=np.arange(1,len(val_gender_accs)+1)
fig,ax=plt.subplots(1,2,figsize=(15,5))
ax=ax.flat
ax[0].plot(epochs,val_gender_accs,label='val_gender_acc')
ax[1].plot(epochs,val_age_maes,label='val_age_mae')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('val_gender_acc')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('val_age_mae')
ax[0].set_title('Validation Gender Accuracy')
ax[1].set_title('Validation Age MAE')
plt.tight_layout()
plt.show()
