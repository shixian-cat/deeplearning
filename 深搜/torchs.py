import torch
X=torch.tensor([[1,2],[3,4],[5,6],[7,8]]).float()
Y=torch.tensor([[3],[7],[11],[15]]).float()

device='cuda' if torch.cuda.is_available() else 'cpu'
X=X.to(device)
Y=Y.to(device)


from torch import nn
class Myneuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer=nn.Linear(2,8)
        self.hidden_layer_to_activation=nn.ReLU()
        self.hidden_to_output_layer=nn.Linear(8,1)
    def forward(self,x):
        x=self.input_to_hidden_layer(x)
        x=self.hidden_layer_to_activation(x)
        x=self.hidden_to_output_layer(x)
        return x
mynet=Myneuralnet().to(device)
loss_func=nn.MSELoss()

_Y=mynet(X)
loss_value=loss_func(_Y,Y)
print(loss_value)

from torch.optim import SGD
opt=SGD(mynet.parameters(),lr=0.001)

loss_history=[]
for i in range(50):
    opt.zero_grad()
    loss_value=loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.item())


import matplotlib.pyplot as plt
plt.plot( loss_history)
plt.xlabel('Eposchs')
plt.ylabel('loss value')
plt.show()