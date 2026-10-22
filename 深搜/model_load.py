import torch


model=get_model()
state_dict=torch.load('E:/denghc/deeplearning/mymodel/test/model.pth')
model.load_state_dict(state_dict)
model.to(device)

save_path='E:/denghc/deeplearning/mymodel/test/model.pth'
torch.save(model.state_dict(),save_path)