from torchvision.datasets import MNIST
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

def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(steps)


import pandas as pd
from pathlib import Path
from PIL import Image

class AutoEncoderDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, image_size: int, train: bool) -> None:
        self.rows = pd.read_csv(csv_path, dtype=str).fillna("").to_dict("records")
        self.image_root = Path(image_root)
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.image_root / Path(row["image_path"]).name
        
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        return {"image": image, "image_path": str(image_path)}


from torch.utils.data import DataLoader

IMG_SIZE = 384
BATCH_SIZE = 16


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder: 384 -> 192 -> 96 -> 48 -> 24 -> 12 -> 6
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 384x384
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                   # 192x192
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 192x192
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                   # 96x96
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 96x96
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                   # 48x48
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 48x48
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                    # 24x24
            # 如果显存够，可以继续往下池化到 12x12 或 6x6
        )
        
        # Decoder: 24 -> 48 -> 96 -> 192 -> 384
        self.decoder = nn.Sequential(
            # 反卷积：公式 (input - 1) * stride - 2*padding + kernel
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 48x48
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 96x96
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 192x192
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),    # 384x384
            nn.Tanh()  # 配合前面的 Normalize 使用
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


def trn(imput,model,cri,opt):
    model.train()
    opt.zero_grad()
    output=model(imput)
    loss=cri(output,imput)
    loss.backward()
    opt.step()
    return loss.item()

def val(imput,model,cri):
    model.eval()
    with torch.no_grad():
        output=model(imput)
        loss=cri(output,imput)
    return loss.item()

def get_model():
    model=ConvAutoEncoder()
    cri=nn.MSELoss()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    return model.to(device),cri,opt

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1) # 限制在 0 到 1 之间

if __name__ == '__main__':

  
    train_ds = AutoEncoderDataset(r"E:\PAS-AGI-Diagnosis\baseline\pas_us_imagelevel_fixed_split_v1\data\splits/train.csv", r"E:\PAS-AGI-Diagnosis\baseline\pas_us_imagelevel_fixed_split_v1\data\images", IMG_SIZE, train=True)
    val_ds = AutoEncoderDataset(r"E:\PAS-AGI-Diagnosis\baseline\pas_us_imagelevel_fixed_split_v1\data\splits/val.csv", r"E:\PAS-AGI-Diagnosis\baseline\pas_us_imagelevel_fixed_split_v1\data\images", IMG_SIZE, train=False)

    trn_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,       
        pin_memory=True      
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    model,cri,opt=get_model()

    n_epochs=20
    trn_losses,val_losses=[],[]
    for epoch in range(n_epochs):
        trn_loss=0
        for i, data in enumerate(trn_dl):
            im = data['image'].to(device) 
            loss=trn(im,model,cri,opt)
            trn_loss+=loss
        trn_loss/=len(trn_dl)
        trn_losses.append(trn_loss)

        val_loss=0
        for i, data in enumerate(val_dl):
            im = data['image'].to(device)
            loss=val(im,model,cri)
            val_loss+=loss
        val_loss/=len(val_dl)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}")

    
    epochs = np.arange(n_epochs)+1
    plt.plot(epochs, trn_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r-', label='Test loss')
    plt.title('Training and Test loss over increasing epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()

    model_path = './pth/pas_convautoencoder.pth'
    torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))

    for _ in range(5):
        ix = np.random.randint(len(val_ds))
        data= val_ds[ix]
        im = data['image'].to(device)
        with torch.no_grad():
            im = model(im[None])[0]
        plt.subplot(121)

        img_show = denormalize(data['image'][0]).detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_show)
        plt.title('input')
        plt.subplot(122)
        img_show = denormalize(data['image'][0]).detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_show)
        plt.title('input')
        plt.subplot(122)
        img_show = denormalize(im[0]).detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_show, cmap='gray')
        plt.title('prediction')
        plt.show()