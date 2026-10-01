import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0,6.0],[7.0,8.0]])
y = torch.tensor([[3],[7],[11],[15]])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
y = y.to(device)

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

    