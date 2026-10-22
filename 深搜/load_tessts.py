from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
  
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn

# X=torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]).float()
# Y=torch.tensor([[3],[7],[11],[15],[19],[23],[27]]).float()

device='cuda' if torch.cuda.is_available() else 'cpu'
# X=X.to(device)
# Y=Y.to(device)
# val_data=torch.tensor([[11,12],[13,14],[15,16],[17,18],[1.5,2.5]]).float().to(device)
# val_targets=torch.tensor([[23],[27],[31],[35],[4]]).float().to(device)

torch.manual_seed(42)
np.random.seed(42)
def generate_synthetic_data(num_samples, num_features=5, noise_std=0.5):
    X = np.random.uniform(-2, 2, size=(num_samples, num_features))
    y_true = (2 * np.sin(X[:, 0]) + 
              0.5 * X[:, 1]**2 + 
              0.3 * X[:, 2] * X[:, 3] + 
              0.1 * np.exp(X[:, 4]))
    noise = np.random.normal(0, noise_std, size=num_samples)
    y = y_true + noise
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) 
    return X_tensor, y_tensor
train_samples = 1000
val_samples = 300

X_trn, y_trn = generate_synthetic_data(train_samples)
X_val, y_val = generate_synthetic_data(val_samples)

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x.clone().detach() # torch.tensor(x).float()
        self.y = y.clone().detach() # torch.tensor(y).float()
    def __len__(self):
        return len(self.x)
    def __getitem__(self,ix):
        return self.x[ix],self.y[ix]
    
trn_ds=MyDataset(X_trn,y_trn)
val_ds=MyDataset(X_val,y_val)
trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False) 


from torch.optim import SGD,Adam
def get_model():
    model=nn.Sequential(
        # nn.Dropout(0.5),
        nn.Linear(5,64),
        nn.ReLU(),
   
        nn.Linear(64,1)
    ).to(device)
    loss_fn=nn.MSELoss()
    optimizer=SGD(model.parameters(),lr=1e-2)
    return model,loss_fn,optimizer

def train_batch(x,y,model,optimizer,loss_fn):
    x=x.to(device)
    y=y.to(device)
    model.train()
    prediction=model(x)
    l1_regulation=0
    for param in model.parameters():
        l1_regulation+=torch.norm(param,1) 
    batch_loss=loss_fn(prediction,y)+0.0001*l1_regulation
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def val_loss(x,y,model,loss_fn):
    x=x.to(device)
    y=y.to(device)
    prediction=model(x)
    val_loss=loss_fn(prediction,y)
    return val_loss.item()

model,loss_fn,optimizer=get_model()
trn_losses,trn_accuracies=[],[]
val_losses,val_accuracies=[],[]

for epoch in range(200):
    print(epoch)
    trn_epoch_losses,trn_epoch_accuracies=[],[]
    val_epoch_losses,val_epoch_accuracies=[],[]

    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        batch_loss=train_batch(x,y,model,optimizer,loss_fn)
        trn_epoch_losses.append(batch_loss)
    
    trn_epoch_loss=np.array(trn_epoch_losses).mean()
    trn_losses.append(trn_epoch_loss)
    for ix,batch in enumerate(iter(val_dl)):
        x,y=batch
        batch_loss=val_loss(x,y,model,loss_fn)
        val_epoch_losses.append(batch_loss)

    val_epoch_loss=np.mean(val_epoch_losses)
    val_losses.append(val_epoch_loss)

save_path='E:/denghc/deeplearning/mymodel/test/model.pth'
torch.save(model.state_dict(),save_path)

print(np.min(trn_losses))
print(np.min(val_losses))  

# prediction=model(X_val.to(device))
# print(prediction)

epochs=np.arange(200)+1
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker
# plt.subplot(121)       
plt.figure(figsize=(16, 8))  # 宽度12英寸，高度5英寸                               # 1行2列的第1个位置
plt.plot(epochs, trn_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')

plt.ylim(0, np.mean(val_losses)*2) 

plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.title('Loss value over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')  
plt.grid('off') 
plt.legend()        
plt.show()

     