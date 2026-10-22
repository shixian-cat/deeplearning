from torchvision import transforms




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
from torch.utils.data import Dataset

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

train_ds = AutoEncoderDataset("路径/train.csv", "路径/images", IMG_SIZE, train=True)
val_ds = AutoEncoderDataset("路径/val.csv", "路径/images", IMG_SIZE, train=False)

train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,       
    pin_memory=True      
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)
