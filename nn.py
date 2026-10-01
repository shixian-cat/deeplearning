import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0,6.0],[7.0,8.0]])
y = torch.tensor([[3],[7],[11],[15]])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
y = y.to(device)
