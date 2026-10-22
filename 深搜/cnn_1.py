import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import Adam
import numpy as np
device = 'cuda' if torch.cuda.is_available()else'cpu'
X_train = torch.tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],
                        [[[-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]]]]
                        ).to(device).float()
X_train /= 8
y_train = torch.tensor([[0],[1]]).to(device).float()

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1, 1),
        nn.Sigmoid()
    ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(),lr=1e-2)
    return model, loss_fn, optimizer
import sys
print(sys.executable)  # 这行会打印出你的代码使用的是哪个 Python 解释器

from torchsummary import summary
model,loss_fn,optimizer= get_model()

def train_batch(x, y, model, optimizer,loss_fn):
    model.train()
    prediction = model(x)
    # print(prediction)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()
trn_dl=DataLoader(TensorDataset(X_train,y_train))

trn_loss=[]
for epoch in range(2000):
    print(epoch)
    trn_epoch_loss=[]
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        batch_loss=train_batch(x,y,model,optimizer,loss_fn)
        trn_epoch_loss.append(batch_loss)
    trn_loss.append(np.mean(trn_epoch_loss))


print(model(X_train[:1]))
epochs=np.arange(2000)+1
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker  
plt.figure(figsize=(16, 8))  # 宽度12英寸，高度5英寸                               # 1行2列的第1个位置
plt.plot(epochs, trn_loss, 'bo', label='Training loss')


plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(50))
plt.title('Loss value over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')  
plt.grid('off') 
plt.legend()        
plt.show()