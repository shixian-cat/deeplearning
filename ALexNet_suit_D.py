import os

import time
import copy
import numpy as np

import matplotlib.pyplot as plt

import torch

import torchvision

import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

ddir = './data/hymenoptera_data'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

img_data ={k: datasets.ImageFolder(os.path.join(ddir, k), data_transforms[k]) for k in ['train', 'val']}
dataloaders = {k: torch.utils.data.DataLoader(img_data[k], batch_size=8, shuffle=True) for k in ['train', 'val']}
dset_sizes = {k: len(img_data[k]) for k in ['train', 'val']}
classes = img_data['train'].classes
print(classes)
dvc=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imageshow(im,text=None):
    im=im.numpy().transpose((1,2,0))
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    im=std*im+mean
    im=np.clip(im,0,1)
    plt.imshow(im)
    if text is not None:
        plt.title(text)

im,cls=next(iter(dataloaders['train']))
grid=torchvision.utils.make_grid(im)
imageshow(grid,text=[classes[x] for x in cls])

def finetune_model(model,loss_f,opt,epo=10):
    start=time.time()
    model_weights=copy.deepcopy(model.state_dict())
    accuracy=0.0
    for e in range(epo):
        print(f'Epoch {e}/{epo-1}')
        print('='*20)

        for dset in['train','val']:
            if dset=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_corrects=0

            for inputs,labels in dataloaders[dset]:
                inputs=inputs.to(dvc)
                labels=labels.to(dvc)

                opt.zero_grad()

                with torch.set_grad_enabled(dset=='train'):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=loss_f(outputs,labels)

                    if dset=='train':
                        loss.backward()
                        opt.step()

                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)

            epoch_loss=running_loss/dset_sizes[dset]
            epoch_acc=running_corrects.double()/dset_sizes[dset]

            print(f'{dset} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if dset=='val' and epoch_acc>accuracy:
                accuracy=epoch_acc
                model_weights=copy.deepcopy(model.state_dict())
        print()

    time_delta=time.time()-start
    print(f'Training complete in {time_delta//60:.0f}m {time_delta%60:.0f}s')
    print(f'Best val Acc: {accuracy:.4f}')

    model.load_state_dict(model_weights)
    return model

def visualize_model(model,num_images=4):
    torch.manual_seed(1)
    was_model_training=model.training
    model.eval()
    im_counter=0
    fig=plt.figure()

    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataloaders['val']):
            inputs=inputs.to(dvc)
            labels=labels.to(dvc)

            outputs=model(inputs)
            _,preds=torch.max(outputs,1)

            for j in range(inputs.size()[0]):
                im_counter+=1
                ax=plt.subplot(num_images//2,2,im_counter)
                ax.axis('off')
                ax.set_title(f'predicted: {classes[preds[j]]}')
                imageshow(inputs.cpu().data[j])

                if im_counter==num_images:
                    model.train(mode=was_model_training)
                    return
        model.train(mode=was_model_training)

model=models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).to(dvc)
model.classifier[6]=nn.Linear(4096, len(classes)).to(dvc)

loss_f=nn.CrossEntropyLoss()
opt=optim.SGD(model.parameters(),lr=0.0001)

model=finetune_model(model,loss_f,opt,epo=10)

visualize_model(model)