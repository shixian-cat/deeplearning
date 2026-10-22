import numpy as np
from copy import deepcopy
x=np.array([[1,1]])
y=np.array([0])

# W =[
#     np.array([[-0.0053, 0.3793],
#               [-0.5820, -0.5204],
#               [-0.2723, 0.1896]], dtype=np.float32).T,
#     np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
#     np.array([[ 0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
#     np.array([-0.5516], dtype=np.float32)
# ]
W =[
    np.array([[0, 0],
              [0, 0],
              [0, 0]], dtype=np.float32).T,
    np.array([0, 0, 0], dtype=np.float32),
    np.array([[ 0, 0, 0.]], dtype=np.float32).T,
    np.array([-0.5516], dtype=np.float32)
]

def feed_forward(input,output,weight):
    pre_hidden=np.dot(input,weight[0])+weight[1]
    hidden=1/(1+np.exp(-pre_hidden))
    out=np.dot(hidden,weight[2])+weight[3]
    mean_squared_error=np.mean(np.square(out-output))
    return mean_squared_error

def update_werghts(inputs,outputs,weights,lr):
    orginal_weights=deepcopy(weights)
    temp_weights=deepcopy(weights)
    updated_werghts=deepcopy(weights)
    orginal_loss=feed_forward(inputs,outputs,weights)
    for i,layer in enumerate(orginal_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights=deepcopy(weights)
            temp_weights[i][index]+=0.0001
            _loss_plus=feed_forward(inputs,outputs,temp_weights)
            grad=(_loss_plus-orginal_loss)/0.0001
            updated_werghts[i][index]-=grad*lr
    return updated_werghts,orginal_loss

losses=[]
for epoch in range(100):
    W,loss=update_werghts(x,y,W,0.01)
    losses.append(loss)
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Eposchs')
plt.ylabel('loss value')
plt.show()