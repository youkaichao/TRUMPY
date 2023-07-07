import torch
from torchvision.models.resnet import resnet50

# define device and models
device = torch.device('cpu')
net = resnet50().to(device)
input = torch.rand(16, 3, 224, 224).to(device)

# if gpu is available, record current memory
if device != torch.device('cpu'):
    current_memory = torch.cuda.memory_allocated()

# define loss function
loss = net(input).sum() # loss is the starting point of the backward computation graph
from trumpy.lib import memory_for_backward
# calculate the memory used for backward
saved_memory = memory_for_backward(net, loss)
print(f'estimated memory usage for backward related tensors in {device}: {saved_memory / 1024 ** 3} GB')

# check how much memory pytorch holds
if device != torch.device('cpu'):
    import gc
    gc.collect()
    saved_memory = torch.cuda.memory_allocated() - current_memory
    print(f'ground-truth memory usage for backward related tensors in gpu: {saved_memory / 1024 ** 3} GB')