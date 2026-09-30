import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = torch.device('cuda')
lenet =LeNet().to(device=device)

def train(net,trainloader,optim,epoch):
    loss_total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optim.step()
        loss_total += loss.item()
        if(i+1) % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss_total / 100))
            loss_total = 0.0

def test(net,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True,download=True, transform=train_transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset=torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=test_transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

OPT=torch.optim.SGD(lenet.parameters(), lr=0.001)
# for epoch in range(50):
#     train(lenet,trainloader,OPT,epoch)
#     print()
#     test(lenet,testloader)
#     print()

model_path = './pth/lenet.pth'
torch.save(lenet.state_dict(), model_path)

d_iter=iter(testloader)
im,lab=next(d_iter)

def imageshow(im,text=None):
    im=im.numpy().transpose((1,2,0))
    mean=np.array([0.5, 0.5, 0.5])
    std=np.array([0.5, 0.5, 0.5])
    im=std*im+mean
    im=np.clip(im,0,1)
    plt.imshow(im)
    if text is not None:
        plt.title(text)

imageshow(torchvision.utils.make_grid(im[:4]))
print('GroundTruth: ', ' '.join('%5s' % classes[lab[j]] for j in range(4)))

lenet_cached=LeNet().to(device=device)
lenet_cached.load_state_dict(torch.load(model_path, weights_only=False))

op=lenet_cached(im.to(device))

_,pred=torch.max(op,1)
print('Predicted: ', ' '.join('%5s' % classes[pred[j]] for j in range(4)))