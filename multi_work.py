import torch
import os
import numpy as np, cv2, pandas as pd, time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trn_df=pd.read_csv('./data/fairface-img-margin025-trainval/fairface_label_train.csv')
val_df=pd.read_csv('./data/fairface-img-margin025-trainval/fairface_label_val.csv')
print(trn_df.head())

IMAGE_SIZE=224
class GenderAgeClass(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].squeeze()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_dir, 'data/fairface-img-margin025-trainval',row.file)
        f = f.replace('\\', '/') 

        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gen = row.gender=='Female'
        
        age_str = str(row['age']).strip()
        if '-' in age_str:
            low, high = age_str.split('-')
            age = (float(low) + float(high)) / 2.0
        elif 'more than' in age_str.lower():
            age = float(age_str.split()[-1]) + 5.0  
        else:
            age = float(age_str)

        return im, age, gen

    def preprocess(self,im):
        im=cv2.resize(im, (IMAGE_SIZE,IMAGE_SIZE))       
        im=torch.tensor(im).permute(2,0,1)
        im=self.normalize(im/255.)
        return im[None]

    def collate_fn(self,batch):
        ims,ages,gens=[],[],[]
        for im,age,gen in batch:
            im=self.preprocess(im)
            ims.append(im)
            ages.append(float(age)/80)
            gens.append(float(gen))

        ages,gens=[torch.tensor(x).float().to(device) for x in [ages,gens]]
        ims= torch.cat(ims).to(device)
        return ims, ages, gens

trn_ds=GenderAgeClass(trn_df)
val_ds=GenderAgeClass(val_df)

trn_ld=DataLoader(trn_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=trn_ds.collate_fn)
val_ld=DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True, collate_fn=val_ds.collate_fn)
a,b,c=next(iter(trn_ld))
print(a.shape,b.shape,c.shape)

def get_model():
    model=models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    model.avgpool= nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )

    class ageGenderClassifier(nn.Module):
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
                nn.ReLU(),
            )
            self.age_classifier=nn.Sequential(
                nn.Linear(64,1),
                nn.Sigmoid()
            )
            self.gender_classifier=nn.Sequential(
                nn.Linear(64,1),
                nn.Sigmoid()
            )

        def forward(self,x):
            x=self.intermediate(x)
            age=self.age_classifier(x)
            gender=self.gender_classifier(x)
            return age,gender
    model.classifier=ageGenderClassifier()
    gender_cri=nn.BCELoss()
    age_cri=nn.MSELoss()
    loss_fun=gender_cri,age_cri
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    return model.to(device), loss_fun, opt

model,loss_fun,opt=get_model()

def trn(data,model,opt,cri):
    model.train()
    ims,age,gen=data
    opt.zero_grad()
    pred_age,pred_gen=model(ims)
    gen_cri,age_cri=cri
    gen_loss=gen_cri(pred_gen.squeeze(),gen)
    age_loss=age_cri(pred_age.squeeze(),age)
    loss=gen_loss+age_loss
    loss.backward()
    opt.step()
    return loss.item()

def val(data,model,opt,cri):
    model.eval()
    ims,age,gen=data
    with torch.no_grad():
        pred_age,pred_gen=model(ims)
        gen_cri,age_cri=cri
        gen_loss=gen_cri(pred_gen.squeeze(),gen)
        age_loss=age_cri(pred_age.squeeze(),age)
        loss=gen_loss+age_loss
        pred_gen_label=(pred_gen.squeeze()>0.5).float()
        gen_acc=(pred_gen_label==gen).float().sum()
        age_mae=(pred_age.squeeze()-age).abs().sum()
    return loss.item(), gen_acc.item(), age_mae.item()

# val_gen_accs,val_age_maes=[],[]
# trn_losses,val_losses=[],[]
# n=10
# best_loss=1000
# start=time.time()
# for epoch in range(n):
#     epoch_trn_loss,epoch_val_loss=0,0
#     val_age_mae, val_gen_acc=0,0
#     _n=len(trn_ld)
#     for i,data in enumerate(trn_ld):
#         loss=trn(data,model,opt,loss_fun)
#         epoch_trn_loss+=loss
#         if i%10==0:
#             print(f'epoch:{epoch}, step:{i}/{_n}, trn_loss:{loss:.4f}')
#     ctr=0
#     for i,data in enumerate(val_ld):
#         loss, gen_acc, age_mae=val(data,model,opt,loss_fun)
#         epoch_val_loss+=loss
#         val_age_mae+=age_mae
#         val_gen_acc+=gen_acc
#         ctr+=len(data[0])

#     val_gen_accs.append(val_gen_acc/ctr)
#     val_age_maes.append(val_age_mae/ctr)
#     trn_losses.append(epoch_trn_loss/_n)
#     val_losses.append(epoch_val_loss/_n)
#     elapsed=time.time()-start
#     print(f'epoch:{epoch}, trn_loss:{trn_losses[-1]:.4f}, val_loss:{val_losses[-1]:.4f}, val_gen_acc:{val_gen_accs[-1]:.4f}, val_age_mae:{val_age_maes[-1]:.4f}, elapsed:{elapsed:.2f}s')

# model_path = './pth/multiwork.pth'
# torch.save(model.state_dict(), model_path)

# epochs = np.arange(1,len(val_gen_accs)+1)
# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax = ax.flat
# ax[0].plot(epochs, val_gen_accs, 'bo')
# ax[1].plot(epochs, val_age_maes, 'r')
# ax[0].set_xlabel('Epochs')
# ax[1].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy')
# ax[1].set_ylabel('MAE')
# ax[0].set_title('Validation Gender Accuracy')
# ax[1].set_title('Validation Age Mean-Absolute-Error')
# plt.show()

model.load_state_dict(torch.load('./pth/multiwork.pth', map_location=device))


im=cv2.imread('./data/fairface-img-margin025-trainval/val/1232.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im=val_ds.preprocess(im).to(device)

age,gen=model(im)
pred_gender=gen.to('cpu').detach().numpy()
pred_age=age.to('cpu').detach().numpy()

im=cv2.imread('./data/fairface-img-margin025-trainval/val/1232.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
print('predicted gender:',np.where(pred_gender[0][0]<0.5, 'Male','Female'), '; Predictedage', int(pred_age[0][0]*80))