import torch
import torch.nn as nn
x = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0,6.0],[7.0,8.0]])
y = torch.tensor([[3],[7],[11],[15]])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
y = y.to(device).float()


from torch import nn
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.linear = nn.Linear(2, 1)  # Input size is 2, output size is 1

        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)
        

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

mynet = MyNeuralNetwork().to(device)
print(mynet.input_to_hidden_layer.weight)

mynet.parameters()
for param in mynet.named_parameters():print(param)

loss_func = nn.MSELoss()
_V = loss_func(mynet(x), y)
print(_V)

from torch import optim
optimizer = optim.SGD(mynet.parameters(), lr=0.01)

loss_history = []
for epoch in range(50):
    optimizer.zero_grad()  # Zero the gradients
    output = mynet(x)      # Forward pass
    loss = loss_func(output, y)  # Compute loss
    loss.backward()        # Backward pass
    optimizer.step()       # Update weights

    loss_history.append(loss.item())  # Store loss value

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
y = [[3], [7], [11], [15]]

x = torch.tensor(x).float()
y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
y = y.to(device)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.clone().detach()  # Ensure x is a tensor and detached from any computation graph
        self.y = y.clone().detach()  # Ensure y is a tensor and detached from any computation graph

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

ds = MyDataset(x, y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

for x, y in dl:
    print(x.cpu(), y.cpu())

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
mynet = MyNeuralNetwork().to(device)
loss_func = nn.MSELoss()
from torch import optim
optimizer = optim.SGD(mynet.parameters(), lr=0.01)

import time
loss_history = []
start = time.time()
for epoch in range(50):
    for data in dl:
        x, y = data
        optimizer.zero_grad()  # Zero the gradients
        loss = loss_func(mynet(x), y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    loss_history.append(loss.item())  # Store loss value

end = time.time()
print(end - start)
    
