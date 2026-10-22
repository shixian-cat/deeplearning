import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
def generate_synthetic_data(num_samples, num_features=5, noise_std=0.3):
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

print("训练集形状:", X_trn.shape, y_trn.shape)   # torch.Size([1000, 5]) torch.Size([1000, 1])
print("验证集形状:", X_val.shape, y_val.shape)       # torch.Size([300, 5])  torch.Size([300, 1])


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

# ---------- 可选：看一眼数据长什么样 ----------
print("\n前3个训练样本:")
for i in range(3):
    print(f"X: {X_trn[i].tolist()}  ->  y: {y_trn[i].item():.3f}")